// SPDX-License-Identifier: AGPL-3.0-only
//! macOS backend: a Seatbelt profile applied by `/usr/bin/sandbox-exec` (`(deny default)`:
//! no network, no files except the system libraries, the program, the read-only paths and the
//! work folder, no signals or process inspection outside the sandbox), rlimits as a backstop,
//! and a parent sampler of the process group (resident memory, CPU time, process count), since
//! macOS does not enforce `RLIMIT_AS`.

use std::ffi::OsString;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use super::{file_sizes, Layer, Outcome, Policy, SandboxError, Support};

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
/// `PROC_RUID_ONLY` from libproc.h.
const PROC_RUID_ONLY: u32 = 5;
/// System folders every program needs to load.
const SYSTEM_READ: [&str; 5] = [
    "/usr/lib",
    "/System",
    "/Library/Apple",
    "/private/var/db/dyld",
    "/dev",
];

pub(super) fn check() -> Support {
    let works = Path::new(SANDBOX_EXEC).is_file()
        && Command::new(SANDBOX_EXEC)
            .args([
                "-p",
                "(version 1)(deny default)(allow process-exec)(allow file-read*)",
                "/usr/bin/true",
            ])
            .env_clear()
            .status()
            .is_ok_and(|s| s.success());
    let note = if works {
        "Seatbelt via sandbox-exec".to_string()
    } else {
        format!("{SANDBOX_EXEC} is missing or cannot apply a profile")
    };
    let layer = |name, mechanism, ok: bool, note: String| Layer {
        name,
        mechanism,
        supported: ok,
        note,
    };
    Support {
        os: "macos",
        layers: vec![
            layer("files", "Seatbelt (deny default)", works, note.clone()),
            layer("network", "Seatbelt (deny default)", works, note.clone()),
            layer(
                "limits",
                "parent sampler + rlimits",
                true,
                "memory, CPU time, processes".into(),
            ),
            layer(
                "file size",
                "RLIMIT_FSIZE + parent watch",
                true,
                "per file".into(),
            ),
            layer(
                "lifetime",
                "process group, killed on return",
                true,
                "a program that calls setsid() leaves the group but keeps every other restriction"
                    .into(),
            ),
            layer(
                "environment",
                "explicit environment",
                true,
                "nothing inherited".into(),
            ),
            layer("other processes", "Seatbelt (deny default)", works, note),
        ],
    }
}

/// A Seatbelt string literal.
fn quote(path: &Path) -> String {
    let s = path.to_string_lossy();
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn canonical(path: &Path) -> Result<PathBuf, SandboxError> {
    std::fs::canonicalize(path).map_err(|e| SandboxError::Policy(format!("{path:?}: {e}")))
}

fn profile(policy: &Policy, program: &Path) -> Result<String, SandboxError> {
    let mut p = String::from(
        "(version 1)\n(deny default)\n(allow process-fork)\n(allow signal (target same-sandbox))\n\
         (allow sysctl-read)\n(allow file-read-metadata)\n",
    );
    let mut read = Vec::new();
    for sys in SYSTEM_READ {
        if Path::new(sys).exists() {
            read.push(format!("(subpath {})", quote(Path::new(sys))));
        }
    }
    let program = canonical(program)?;
    read.push(format!("(literal {})", quote(&program)));
    let mut exec = vec![format!("(literal {})", quote(&program))];
    for ro in &policy.read_only {
        let ro = canonical(ro)?;
        let filter = if ro.is_dir() { "subpath" } else { "literal" };
        read.push(format!("({filter} {})", quote(&ro)));
        exec.push(format!("({filter} {})", quote(&ro)));
    }
    let work = canonical(&policy.work_dir)?;
    p += &format!("(allow file-read* {})\n", read.join(" "));
    p += &format!("(allow process-exec {})\n", exec.join(" "));
    p += &format!(
        "(allow file-read* file-write* (subpath {}))\n",
        quote(&work)
    );
    p += "(allow file-write-data (literal \"/dev/null\"))\n";
    Ok(p)
}

/// Processes owned by this real user, for RLIMIT_NPROC, which counts them all.
fn user_processes() -> u64 {
    // SAFETY: a null buffer asks for the needed size; getuid cannot fail.
    let n = unsafe { libc::proc_listpids(PROC_RUID_ONLY, libc::getuid(), std::ptr::null_mut(), 0) };
    (n.max(0) as u64) / std::mem::size_of::<libc::pid_t>() as u64
}

/// Resident bytes, CPU seconds and process count of every process in group `pgid`.
#[allow(deprecated)] // mach_timebase_info: libc points to the mach2 crate; one call is enough.
fn sample_group(pgid: i32) -> (u64, f64, u32) {
    let mut pids = vec![0 as libc::pid_t; 4096];
    let bytes = (pids.len() * std::mem::size_of::<libc::pid_t>()) as libc::c_int;
    // SAFETY: the buffer holds `bytes` bytes.
    let n = unsafe { libc::proc_listpgrppids(pgid, pids.as_mut_ptr().cast(), bytes) };
    let count = (n.max(0) as usize / std::mem::size_of::<libc::pid_t>()).min(pids.len());
    let mut tb = libc::mach_timebase_info { numer: 1, denom: 1 };
    // SAFETY: valid out-pointer.
    unsafe { libc::mach_timebase_info(&mut tb) };
    let ticks_to_s = f64::from(tb.numer) / f64::from(tb.denom.max(1)) / 1e9;
    let (mut rss, mut cpu, mut live) = (0u64, 0f64, 0u32);
    for &pid in &pids[..count] {
        if pid <= 0 {
            continue;
        }
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        // SAFETY: `info` is `size` bytes.
        let got = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTASKINFO,
                0,
                (&mut info as *mut libc::proc_taskinfo).cast(),
                size,
            )
        };
        if got == size {
            live += 1;
            rss += info.pti_resident_size;
            cpu += (info.pti_total_user + info.pti_total_system) as f64 * ticks_to_s;
        }
    }
    (rss, cpu, live)
}

pub(super) fn run(
    policy: &Policy,
    program: &Path,
    args: &[OsString],
    stdin: &[u8],
) -> Result<Outcome, SandboxError> {
    let before = file_sizes(&policy.work_dir);
    let profile = profile(policy, program)?;
    let program = canonical(program)?;
    let cpu = policy.cpu_time.as_secs().max(1);
    let nproc = user_processes() + u64::from(policy.max_processes);
    let fsize = policy.max_file_bytes;

    let mut cmd = Command::new(SANDBOX_EXEC);
    cmd.arg("-p")
        .arg(&profile)
        .arg(&program)
        .args(args)
        .env_clear()
        .envs(policy.env.iter().map(|(k, v)| (k, v)))
        .current_dir(&policy.work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    // SAFETY: runs between fork and exec and only calls setrlimit.
    unsafe {
        cmd.pre_exec(move || {
            let limits = [
                (libc::RLIMIT_CPU, cpu + 1, cpu + 2),
                (libc::RLIMIT_NPROC, nproc, nproc),
                (libc::RLIMIT_FSIZE, fsize, fsize),
                (libc::RLIMIT_CORE, 0, 0),
            ];
            for (res, soft, hard) in limits {
                let lim = libc::rlimit {
                    rlim_cur: soft as libc::rlim_t,
                    rlim_max: hard as libc::rlim_t,
                };
                if libc::setrlimit(res, &lim) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let started = Instant::now();
    let child = cmd
        .spawn()
        .map_err(|e| SandboxError::Os(format!("starting {program:?} failed: {e}")))?;
    super::unix::supervise(child, started, policy, stdin, &before, sample_group)
}
