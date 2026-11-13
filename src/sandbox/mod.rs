// SPDX-License-Identifier: AGPL-3.0-only
//! Developer sandbox (roadmap v0.2.7.6, THME-3020). Design: docs/sandbox.md.
//!
//! [`run`] starts one program as an isolated process: no network, write access only to its work
//! folder, read access only to the paths it is given, and caps on wall time, CPU time, memory,
//! processes, file size and output. Nothing is inherited from the caller: no environment, no
//! handles other than the three pipes.
//!
//! **Fail closed.** If this host cannot enforce every layer, [`run`] returns
//! [`SandboxError::Unsupported`] and starts nothing. There is no flag that turns layers off.
//! [`check`] reports, layer by layer, what this host can enforce.
//!
//! Backends: Windows (AppContainer and job object), Linux (Landlock, seccomp, rlimits). macOS is
//! refused until its backend lands; see the roadmap.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(target_os = "linux")]
mod linux;
pub mod probe;
#[cfg(windows)]
mod windows;

/// What the sandboxed program may do. Network access is never granted.
#[derive(Debug, Clone)]
pub struct Policy {
    /// The only writable place; also the working directory.
    pub work_dir: PathBuf,
    /// Readable paths (files or folders), for example an interpreter and its libraries. The
    /// program itself is always readable.
    pub read_only: Vec<PathBuf>,
    /// The whole environment. Nothing else is inherited. On Windows `SystemRoot` is added, and
    /// `LOCALAPPDATA`, `TEMP` and `TMP` point to the run's private container folder.
    pub env: Vec<(String, String)>,
    pub wall_time: Duration,
    pub cpu_time: Duration,
    pub memory_bytes: u64,
    pub max_processes: u32,
    /// Largest file the program may write in `work_dir`.
    pub max_file_bytes: u64,
    /// Cap on stdout and on stderr, each.
    pub max_output_bytes: u64,
}

impl Policy {
    /// Conservative defaults around a work folder: 60 s wall, 30 s CPU, 1 GiB memory,
    /// 16 processes, 256 MiB per file, 16 MiB of output.
    pub fn new(work_dir: impl Into<PathBuf>) -> Self {
        Self {
            work_dir: work_dir.into(),
            read_only: Vec::new(),
            env: Vec::new(),
            wall_time: Duration::from_secs(60),
            cpu_time: Duration::from_secs(30),
            memory_bytes: 1 << 30,
            max_processes: 16,
            max_file_bytes: 256 << 20,
            max_output_bytes: 16 << 20,
        }
    }

    fn validate(&self) -> Result<(), SandboxError> {
        let bad = |m: String| Err(SandboxError::Policy(m));
        if !self.work_dir.is_dir() {
            return bad(format!("work_dir {:?} is not a folder", self.work_dir));
        }
        for p in &self.read_only {
            if !p.exists() {
                return bad(format!("read-only path {p:?} does not exist"));
            }
        }
        if self.wall_time.is_zero() || self.cpu_time.is_zero() {
            return bad("time limits must be above zero".into());
        }
        if self.memory_bytes < (16 << 20) {
            return bad("memory_bytes must be at least 16 MiB".into());
        }
        if self.max_processes == 0 {
            return bad("max_processes must be at least 1".into());
        }
        for (k, _) in &self.env {
            if k.is_empty() || k.contains('=') || k.contains('\0') {
                return bad(format!("invalid environment name {k:?}"));
            }
        }
        Ok(())
    }
}

/// How a sandboxed run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    /// The program exited with this code.
    Exited(i32),
    /// The program was killed by this signal (Unix).
    Signalled(i32),
    TimeLimit,
    CpuLimit,
    MemoryLimit,
    /// Output or file-size cap reached.
    OutputLimit,
    /// Process cap reached; the run was stopped.
    ProcessLimit,
}

/// The result of a sandboxed run. When it is returned, every process of the run is gone.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub ended: Ended,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub wall: Duration,
}

#[derive(Debug)]
pub enum SandboxError {
    /// This host cannot enforce `layer`; nothing was started.
    Unsupported { layer: &'static str, reason: String },
    /// The policy itself is invalid.
    Policy(String),
    /// An OS call failed while setting up or running.
    Os(String),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { layer, reason } => {
                write!(
                    f,
                    "sandbox unsupported on this host ({layer}): {reason}; nothing was started"
                )
            }
            Self::Policy(m) => write!(f, "invalid sandbox policy: {m}"),
            Self::Os(m) => write!(f, "sandbox error: {m}"),
        }
    }
}

impl std::error::Error for SandboxError {}

/// One isolation layer and whether this host enforces it.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: &'static str,
    pub mechanism: &'static str,
    pub supported: bool,
    pub note: String,
}

/// What this host can enforce.
#[derive(Debug, Clone)]
pub struct Support {
    pub os: &'static str,
    pub layers: Vec<Layer>,
}

impl Support {
    pub fn all_supported(&self) -> bool {
        !self.layers.is_empty() && self.layers.iter().all(|l| l.supported)
    }

    fn first_missing(&self) -> Option<SandboxError> {
        if self.layers.is_empty() {
            return Some(SandboxError::Unsupported {
                layer: "backend",
                reason: format!("no sandbox backend for {} yet", self.os),
            });
        }
        self.layers
            .iter()
            .find(|l| !l.supported)
            .map(|l| SandboxError::Unsupported {
                layer: l.name,
                reason: l.note.clone(),
            })
    }
}

/// Reports which layers this host can enforce.
pub fn check() -> Support {
    #[cfg(windows)]
    {
        windows::check()
    }
    #[cfg(target_os = "linux")]
    {
        linux::check()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Support {
            os: std::env::consts::OS,
            layers: Vec::new(),
        }
    }
}

/// Runs `program` with `args` under `policy`, feeding `stdin`. See the module documentation.
pub fn run(
    policy: &Policy,
    program: &Path,
    args: &[OsString],
    stdin: &[u8],
) -> Result<Outcome, SandboxError> {
    policy.validate()?;
    if !program.is_file() {
        return Err(SandboxError::Policy(format!(
            "program {program:?} is not a file"
        )));
    }
    if let Some(missing) = check().first_missing() {
        return Err(missing);
    }
    #[cfg(windows)]
    {
        windows::run(policy, program, args, stdin)
    }
    #[cfg(target_os = "linux")]
    {
        linux::run(policy, program, args, stdin)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = (args, stdin);
        Err(SandboxError::Unsupported {
            layer: "backend",
            reason: format!("no sandbox backend for {} yet", std::env::consts::OS),
        })
    }
}

/// Collects a pipe into memory up to `cap` bytes; returns the bytes and whether the cap was hit.
/// Keeps reading (and discarding) after the cap is hit so the writer is never blocked; the
/// caller stops the run when it sees the flag.
#[cfg_attr(not(any(windows, target_os = "linux")), allow(dead_code))]
fn drain(
    mut reader: impl std::io::Read,
    cap: u64,
    over: &std::sync::atomic::AtomicBool,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) | Err(_) => return out,
            Ok(n) => {
                let room = (cap as usize).saturating_sub(out.len());
                out.extend_from_slice(&buf[..n.min(room)]);
                if n > room {
                    over.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
    }
}

/// Sizes of every file under `dir`, recursively, without following links.
#[cfg_attr(not(any(windows, target_os = "linux")), allow(dead_code))]
fn file_sizes(dir: &Path) -> std::collections::HashMap<PathBuf, u64> {
    let mut sizes = std::collections::HashMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let Ok(meta) = e.path().symlink_metadata() else {
                continue;
            };
            if meta.is_dir() {
                stack.push(e.path());
            } else {
                sizes.insert(e.path(), meta.len());
            }
        }
    }
    sizes
}

/// Whether the run wrote a file over `cap`: a file that is new or grew since `before` and is
/// now larger than `cap`. Files that were already there unchanged do not count.
#[cfg_attr(not(any(windows, target_os = "linux")), allow(dead_code))]
fn wrote_over(dir: &Path, before: &std::collections::HashMap<PathBuf, u64>, cap: u64) -> bool {
    file_sizes(dir)
        .into_iter()
        .any(|(p, size)| size > cap && before.get(&p).is_none_or(|&b| size > b))
}
