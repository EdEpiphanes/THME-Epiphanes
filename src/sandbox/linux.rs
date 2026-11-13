// SPDX-License-Identifier: AGPL-3.0-only
//! Linux backend. Layers, applied in the child between fork and exec:
//!
//! - **Landlock** (ABI 3, Linux 6.2, or newer): read and execute only in the system library
//!   folders, the program and the policy's read-only paths; write only in the work folder.
//!   TCP rules (ABI 4) and signal and abstract-socket scoping (ABI 6) are added where available.
//! - **seccomp**: `socket()` refused for every family, `io_uring` refused (its socket operations
//!   would bypass the filter), no `ptrace` or cross-process memory access, no leaving the
//!   process group, no keyrings or namespaces, and no signals to processes by pid.
//! - **rlimits** as a backstop: CPU, address space (twice the memory cap), file size, core
//!   dumps, process count.
//!
//! The parent samples the process group every 50 ms (resident memory, CPU time, process count),
//! so limits are reported the same way as on Windows, and kills the whole group on return.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use landlock::{
    Access, AccessFs, AccessNet, CompatLevel, Compatible, PathBeneath, PathFd, Ruleset,
    RulesetAttr, RulesetCreatedAttr, Scope, ABI,
};
use seccompiler::{
    BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
    SeccompRule,
};

use super::{file_sizes, Layer, Outcome, Policy, SandboxError, Support};

/// Lowest Landlock ABI accepted: 3 adds `Truncate`, without which a file outside the work folder
/// could still be truncated.
const MIN_ABI: i32 = 3;

/// System folders every program needs to load (dynamic loader, C library, interpreters).
const SYSTEM_READ: [&str; 6] = [
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/etc/ld.so.cache",
    "/dev/urandom",
];

fn landlock_abi() -> i32 {
    // SAFETY: LANDLOCK_CREATE_RULESET_VERSION (1) with no attributes only queries the ABI.
    let v = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            1u32,
        )
    };
    if v < 0 {
        0
    } else {
        v as i32
    }
}

fn abi_from(v: i32) -> ABI {
    match v {
        ..=0 => ABI::Unsupported,
        1 => ABI::V1,
        2 => ABI::V2,
        3 => ABI::V3,
        4 => ABI::V4,
        5 => ABI::V5,
        _ => ABI::V6,
    }
}

fn seccomp_available() -> bool {
    // SAFETY: PR_GET_SECCOMP takes no pointer; it fails with EINVAL without seccomp support.
    unsafe { libc::prctl(libc::PR_GET_SECCOMP, 0, 0, 0, 0) >= 0 }
}

pub(super) fn check() -> Support {
    let abi = landlock_abi();
    let sec = seccomp_available();
    let files_note = if abi >= MIN_ABI {
        format!("Landlock ABI {abi}")
    } else if abi == 0 {
        "Landlock is not enabled in this kernel".to_string()
    } else {
        format!("Landlock ABI {abi}; ABI {MIN_ABI} (Linux 6.2) or newer is required")
    };
    let sec_note = if sec {
        "seccomp filter"
    } else {
        "seccomp is not available"
    };
    let signals = if abi >= 6 {
        "Landlock signal scoping; seccomp refuses ptrace and kill by pid".to_string()
    } else {
        "seccomp refuses ptrace, kill and sigqueue by pid; thread signals (tgkill) are not filtered below Landlock ABI 6".to_string()
    };
    let layer = |name, mechanism, ok: bool, note: String| Layer {
        name,
        mechanism,
        supported: ok,
        note,
    };
    Support {
        os: "linux",
        layers: vec![
            layer("files", "Landlock", abi >= MIN_ABI, files_note),
            layer(
                "network",
                "seccomp: no socket(), no io_uring",
                sec,
                sec_note.into(),
            ),
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
                "process group; seccomp refuses setsid and setpgid",
                sec,
                sec_note.into(),
            ),
            layer(
                "environment",
                "explicit environment",
                true,
                "nothing inherited".into(),
            ),
            layer(
                "other processes",
                "seccomp (+ Landlock scoping)",
                sec,
                signals,
            ),
        ],
    }
}

fn landlock_ruleset(policy: &Policy, program: &Path) -> Result<OwnedFd, SandboxError> {
    let v = landlock_abi();
    let abi = abi_from(v);
    let err = |e: &dyn std::fmt::Display| SandboxError::Os(format!("Landlock: {e}"));
    let read_dir = AccessFs::from_read(abi);
    let read_file = AccessFs::from_read(abi) & AccessFs::from_file(abi);
    let all = AccessFs::from_all(abi);
    let mut ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_all(ABI::V3))
        .map_err(|e| err(&e))?
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| err(&e))?;
    if v >= 4 {
        ruleset = ruleset
            .handle_access(AccessNet::from_all(abi))
            .map_err(|e| err(&e))?;
    }
    if v >= 6 {
        ruleset = ruleset.scope(Scope::from_all(abi)).map_err(|e| err(&e))?;
    }
    let mut created = ruleset.create().map_err(|e| err(&e))?;
    let mut rule = |path: &Path, access| -> Result<(), SandboxError> {
        let fd = PathFd::new(path).map_err(|e| err(&e))?;
        (&mut created)
            .add_rule(PathBeneath::new(fd, access))
            .map_err(|e| err(&e))?;
        Ok(())
    };
    let pick = |p: &Path| if p.is_dir() { read_dir } else { read_file };
    for sys in SYSTEM_READ {
        let p = Path::new(sys);
        if p.exists() {
            rule(p, pick(p))?;
        }
    }
    let dev_null = Path::new("/dev/null");
    rule(
        dev_null,
        AccessFs::from_file(abi) & (AccessFs::ReadFile | AccessFs::WriteFile),
    )?;
    rule(program, read_file)?;
    for p in &policy.read_only {
        rule(p, pick(p))?;
    }
    rule(&policy.work_dir, all)?;
    let fd: Option<OwnedFd> = created.into();
    fd.ok_or_else(|| SandboxError::Unsupported {
        layer: "files",
        reason: "Landlock ruleset could not be created".into(),
    })
}

fn seccomp_program() -> Result<BpfProgram, SandboxError> {
    let err = |e: &dyn std::fmt::Display| SandboxError::Os(format!("seccomp: {e}"));
    // Signals and queued signals to a process by pid (> 0) are refused; 0 and negative values
    // (own process group) stay allowed.
    let by_pid = || -> Result<Vec<SeccompRule>, SandboxError> {
        Ok(vec![SeccompRule::new(vec![
            SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Gt, 0)
                .map_err(|e| err(&e))?,
            SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Lt, 0x8000_0000)
                .map_err(|e| err(&e))?,
        ])
        .map_err(|e| err(&e))?])
    };
    let always: [libc::c_long; 20] = [
        libc::SYS_socket,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_setsid,
        libc::SYS_setpgid,
        libc::SYS_pidfd_open,
        libc::SYS_pidfd_send_signal,
        libc::SYS_pidfd_getfd,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_mount,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
    ];
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
    for nr in always {
        rules.insert(nr, Vec::new());
    }
    for nr in [libc::SYS_kill, libc::SYS_rt_sigqueueinfo] {
        rules.insert(nr, by_pid()?);
    }
    #[cfg(target_arch = "x86_64")]
    {
        // The x32 ABI reaches the same calls with bit 30 set; refuse those numbers too.
        let x32: Vec<(i64, Vec<SeccompRule>)> = rules
            .iter()
            .map(|(nr, r)| (nr | 0x4000_0000, r.clone()))
            .collect();
        rules.extend(x32);
    }
    #[cfg(target_arch = "x86_64")]
    let arch = seccompiler::TargetArch::x86_64;
    #[cfg(target_arch = "aarch64")]
    let arch = seccompiler::TargetArch::aarch64;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return Err(SandboxError::Unsupported {
        layer: "network",
        reason: "seccomp filter is built for x86_64 and aarch64 only".into(),
    });
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let filter = SeccompFilter::new(
            rules,
            SeccompAction::Allow,
            SeccompAction::Errno(libc::EPERM as u32),
            arch,
        )
        .map_err(|e| err(&e))?;
        BpfProgram::try_from(filter).map_err(|e| err(&e))
    }
}

/// Tasks (threads) owned by this user, for RLIMIT_NPROC, which counts every task of the user.
fn user_tasks() -> u64 {
    // SAFETY: getuid cannot fail.
    let uid = unsafe { libc::getuid() }.to_string();
    let mut n = 0;
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return 0;
    };
    for e in dir.flatten() {
        let Ok(status) = std::fs::read_to_string(e.path().join("status")) else {
            continue;
        };
        let mine = status
            .lines()
            .any(|l| l.starts_with("Uid:") && l.split_whitespace().nth(1) == Some(uid.as_str()));
        if mine {
            n += status
                .lines()
                .find_map(|l| l.strip_prefix("Threads:"))
                .and_then(|t| t.trim().parse::<u64>().ok())
                .unwrap_or(1);
        }
    }
    n
}

/// Resident bytes, CPU seconds and process count of every process in group `pgid`.
fn sample_group(pgid: i32) -> (u64, f64, u32) {
    // SAFETY: sysconf cannot fail for these names.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as u64;
    let tick = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f64;
    let (mut rss, mut cpu, mut count) = (0u64, 0f64, 0u32);
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return (0, 0.0, 0);
    };
    for e in dir.flatten() {
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else {
            continue;
        };
        // Fields after the command name, which is in parentheses and may contain spaces.
        let Some(rest) = stat.rfind(')').map(|i| &stat[i + 2..]) else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        // f[0] = state (field 3), f[2] = pgrp (5), f[11] = utime (14), f[12] = stime (15),
        // f[21] = rss (24).
        if f.len() < 22 || f[2].parse::<i32>().ok() != Some(pgid) || f[0] == "Z" {
            continue;
        }
        count += 1;
        cpu += (f[11].parse::<f64>().unwrap_or(0.0) + f[12].parse::<f64>().unwrap_or(0.0)) / tick;
        rss += f[21].parse::<u64>().unwrap_or(0) * page;
    }
    (rss, cpu, count)
}

pub(super) fn run(
    policy: &Policy,
    program: &Path,
    args: &[OsString],
    stdin: &[u8],
) -> Result<Outcome, SandboxError> {
    let before = file_sizes(&policy.work_dir);
    let ruleset = landlock_ruleset(policy, program)?;
    let bpf = seccomp_program()?;
    let ruleset_fd = ruleset.as_raw_fd();
    let cpu = policy.cpu_time.as_secs().max(1);
    let address_space = policy.memory_bytes.saturating_mul(2);
    let nproc = user_tasks() + u64::from(policy.max_processes);
    let fsize = policy.max_file_bytes;

    let mut cmd = Command::new(program);
    cmd.args(args)
        .env_clear()
        .envs(policy.env.iter().map(|(k, v)| (k, v)))
        .current_dir(&policy.work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    // SAFETY: the closure runs in the child between fork and exec and only makes system calls
    // (no allocation): prctl, setrlimit, landlock_restrict_self and seccomp.
    unsafe {
        cmd.pre_exec(move || {
            let fail = || Err(std::io::Error::last_os_error());
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) != 0 {
                return fail();
            }
            let limits = [
                (libc::RLIMIT_CPU, cpu + 1, cpu + 2),
                (libc::RLIMIT_AS, address_space, address_space),
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
                    return fail();
                }
            }
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return fail();
            }
            if libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0u32) != 0 {
                return fail();
            }
            seccompiler::apply_filter(&bpf).map_err(|e| std::io::Error::other(e.to_string()))
        });
    }
    let started = Instant::now();
    let child = cmd
        .spawn()
        .map_err(|e| SandboxError::Os(format!("starting {program:?} failed: {e}")))?;
    drop(ruleset);
    super::unix::supervise(child, started, policy, stdin, &before, sample_group)
}
