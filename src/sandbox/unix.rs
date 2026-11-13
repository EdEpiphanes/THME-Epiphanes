// SPDX-License-Identifier: AGPL-3.0-only
//! Supervision shared by the Linux and macOS backends: feed stdin, collect capped output,
//! sample the process group, enforce the caps, and kill the whole group on return.

use std::collections::HashMap;
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{drain, wrote_over, Ended, Outcome, Policy, SandboxError};

/// One sample of a process group: resident bytes, CPU seconds, live processes.
pub(super) type Sampler = fn(i32) -> (u64, f64, u32);

/// Supervises `child`, started at `started` as the leader of its own process group.
pub(super) fn supervise(
    mut child: Child,
    started: Instant,
    policy: &Policy,
    stdin: &[u8],
    before: &HashMap<PathBuf, u64>,
    sample: Sampler,
) -> Result<Outcome, SandboxError> {
    let pgid = child.id() as i32;
    let kill_group = || {
        // SAFETY: signals only the run's own process group.
        unsafe { libc::killpg(pgid, libc::SIGKILL) };
    };
    let over = Arc::new(AtomicBool::new(false));
    let cap = policy.max_output_bytes;
    let (o1, o2) = (over.clone(), over.clone());
    let out_pipe = child.stdout.take();
    let err_pipe = child.stderr.take();
    let out_thread =
        std::thread::spawn(move || out_pipe.map(|p| drain(p, cap, &o1)).unwrap_or_default());
    let err_thread =
        std::thread::spawn(move || err_pipe.map(|p| drain(p, cap, &o2)).unwrap_or_default());
    let mut in_pipe = child.stdin.take();
    let input = stdin.to_vec();
    let in_thread = std::thread::spawn(move || {
        if let Some(p) = in_pipe.as_mut() {
            let _ = p.write_all(&input);
        }
    });

    let cpu_cap = policy.cpu_time.as_secs_f64();
    let file_over = || wrote_over(&policy.work_dir, before, policy.max_file_bytes);
    let mut ended = None;
    let status = loop {
        let exited = child
            .try_wait()
            .map_err(|e| SandboxError::Os(format!("waiting failed: {e}")))?;
        let (rss, cpu_used, procs) = sample(pgid);
        let reason = if rss > policy.memory_bytes {
            Some(Ended::MemoryLimit)
        } else if procs >= policy.max_processes {
            Some(Ended::ProcessLimit)
        } else if cpu_used > cpu_cap {
            Some(Ended::CpuLimit)
        } else if over.load(Ordering::SeqCst) || file_over() {
            Some(Ended::OutputLimit)
        } else if started.elapsed() >= policy.wall_time {
            Some(Ended::TimeLimit)
        } else {
            None
        };
        if reason.is_some() && exited.is_none() {
            ended = reason;
            kill_group();
            break child.wait().ok();
        }
        if let Some(status) = exited {
            break Some(status);
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let wall = started.elapsed();
    kill_group();
    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();
    let _ = in_thread.join();
    let ended = ended.unwrap_or_else(|| match status {
        _ if over.load(Ordering::SeqCst) || file_over() => Ended::OutputLimit,
        Some(s) => match (s.code(), s.signal()) {
            (Some(c), _) => Ended::Exited(c),
            (None, Some(libc::SIGXCPU)) => Ended::CpuLimit,
            (None, Some(libc::SIGXFSZ)) => Ended::OutputLimit,
            (None, Some(sig)) => Ended::Signalled(sig),
            (None, None) => Ended::Signalled(0),
        },
        None => Ended::Signalled(libc::SIGKILL),
    });
    Ok(Outcome {
        ended,
        stdout,
        stderr,
        wall,
    })
}
