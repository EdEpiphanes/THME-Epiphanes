// SPDX-License-Identifier: AGPL-3.0-only
//! Escape tests for the developer sandbox (docs/sandbox.md, E1-E13). Each test runs the probe
//! (`thme sandbox probe`) inside the sandbox, attempts one forbidden action and asserts that it
//! failed. Probe exit 0 means the action succeeded. On an OS without a backend yet, `run` must
//! refuse to start anything.

// Most helpers serve the Windows suite; other OSes only check the refusal until their backends land.
#![cfg_attr(not(windows), allow(dead_code, unused_imports))]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use thme::sandbox::{run, Ended, Outcome, Policy, SandboxError};

const DID: i32 = 0;

/// A temporary layout: `bin/` with a copy of the probe, `work/`, and a canary outside `work/`.
struct Rig {
    root: tempfile::TempDir,
    probe: PathBuf,
}

impl Rig {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("bin")).unwrap();
        std::fs::create_dir(root.path().join("work")).unwrap();
        std::fs::write(root.path().join("canary.txt"), b"secret").unwrap();
        let exe = PathBuf::from(env!("CARGO_BIN_EXE_thme"));
        let probe = root.path().join("bin").join(exe.file_name().unwrap());
        std::fs::copy(&exe, &probe).unwrap();
        Self { root, probe }
    }

    fn work(&self) -> PathBuf {
        self.root.path().join("work")
    }

    fn outside(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    fn policy(&self) -> Policy {
        let mut p = Policy::new(self.work());
        p.wall_time = Duration::from_secs(20);
        p.cpu_time = Duration::from_secs(10);
        p.memory_bytes = 512 << 20;
        p.max_processes = 8;
        p.max_file_bytes = 8 << 20;
        p.max_output_bytes = 1 << 20;
        p
    }

    fn probe(&self, policy: &Policy, args: &[&str]) -> Result<Outcome, SandboxError> {
        let mut all: Vec<OsString> = vec!["sandbox".into(), "probe".into()];
        all.extend(args.iter().map(OsString::from));
        run(policy, &self.probe, &all, &[])
    }
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[cfg(not(windows))]
#[test]
fn refuses_to_start_without_a_backend() {
    let rig = Rig::new();
    match rig.probe(&rig.policy(), &["exit", "0"]) {
        Err(SandboxError::Unsupported { .. }) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert!(!thme::sandbox::check().all_supported());
}

#[cfg(windows)]
mod windows {
    use super::*;

    fn run_ok(rig: &Rig, policy: &Policy, args: &[&str]) -> Outcome {
        rig.probe(policy, args)
            .unwrap_or_else(|e| panic!("{args:?}: {e}"))
    }

    /// The attempt must not have succeeded.
    fn assert_blocked(o: &Outcome, what: &str) {
        assert_ne!(
            o.ended,
            Ended::Exited(DID),
            "{what} escaped: {}",
            String::from_utf8_lossy(&o.stdout)
        );
    }

    #[test]
    fn host_supports_every_layer() {
        assert!(thme::sandbox::check().all_supported());
    }

    #[test]
    fn e01_reading_outside_the_work_folder_is_denied() {
        let rig = Rig::new();
        let o = run_ok(
            &rig,
            &rig.policy(),
            &["read", s(&rig.outside("canary.txt"))],
        );
        assert_blocked(&o, "read canary");
    }

    #[test]
    fn e02_writing_outside_and_through_links_is_denied() {
        let rig = Rig::new();
        let p = rig.policy();
        let canary = rig.outside("canary.txt");
        let o = run_ok(&rig, &p, &["write", s(&rig.outside("new.txt"))]);
        assert_blocked(&o, "write outside");
        assert!(!rig.outside("new.txt").exists());
        let hl = rig.work().join("hl.txt");
        let o = run_ok(&rig, &p, &["hardlink", s(&canary), s(&hl)]);
        assert_blocked(&o, "hard link");
        let sl = rig.work().join("sl.txt");
        let o = run_ok(&rig, &p, &["symlink", s(&canary), s(&sl)]);
        assert_blocked(&o, "symbolic link");
        assert_eq!(std::fs::read(&canary).unwrap(), b"secret");
    }

    #[test]
    fn e03_renaming_out_of_the_work_folder_is_denied() {
        let rig = Rig::new();
        let inside = rig.work().join("m.txt");
        std::fs::write(&inside, b"x").unwrap();
        let dest = rig.work().join("..").join("moved.txt");
        let o = run_ok(&rig, &rig.policy(), &["rename", s(&inside), s(&dest)]);
        assert_blocked(&o, "rename out");
        assert!(inside.exists() && !rig.outside("moved.txt").exists());
    }

    #[test]
    fn e04_network_and_loopback_are_unreachable() {
        use std::net::{TcpListener, UdpSocket};
        let rig = Rig::new();
        let p = rig.policy();
        let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
        tcp.set_nonblocking(true).unwrap();
        let udp = UdpSocket::bind("127.0.0.1:0").unwrap();
        udp.set_nonblocking(true).unwrap();
        let t = tcp.local_addr().unwrap().to_string();
        let u = udp.local_addr().unwrap().to_string();
        assert_blocked(&run_ok(&rig, &p, &["tcp", &t]), "tcp loopback");
        assert_blocked(&run_ok(&rig, &p, &["udp", &u]), "udp loopback");
        assert!(tcp.accept().is_err(), "listener accepted a connection");
        let mut buf = [0u8; 64];
        assert!(
            udp.recv_from(&mut buf).is_err(),
            "listener received a datagram"
        );
        // Routable addresses and name lookup; they fail whether or not CI has internet.
        assert_blocked(&run_ok(&rig, &p, &["tcp", "1.1.1.1:443"]), "tcp out");
        assert_blocked(&run_ok(&rig, &p, &["udp", "1.1.1.1:53"]), "udp out");
        assert_blocked(&run_ok(&rig, &p, &["dns", "example.com"]), "dns");
    }

    #[test]
    fn e06_memory_cap() {
        let rig = Rig::new();
        let o = run_ok(&rig, &rig.policy(), &["alloc", "2000000000"]);
        assert_blocked(&o, "alloc");
        assert_eq!(o.ended, Ended::MemoryLimit);
    }

    #[test]
    fn e07_cpu_and_wall_caps() {
        let rig = Rig::new();
        let mut p = rig.policy();
        p.cpu_time = Duration::from_secs(1);
        p.wall_time = Duration::from_secs(30);
        let o = run_ok(&rig, &p, &["spin", "60"]);
        assert_eq!(o.ended, Ended::CpuLimit, "{o:?}");
        let mut p = rig.policy();
        p.wall_time = Duration::from_secs(2);
        let o = run_ok(&rig, &p, &["sleep", "60"]);
        assert_eq!(o.ended, Ended::TimeLimit);
        assert!(o.wall < Duration::from_secs(10));
    }

    #[test]
    fn e08_process_cap() {
        let rig = Rig::new();
        let o = run_ok(&rig, &rig.policy(), &["spawn", "100"]);
        assert_blocked(&o, "spawn 100");
        let out = String::from_utf8_lossy(&o.stdout);
        assert_eq!(o.ended, Ended::ProcessLimit, "{out}");
    }

    #[test]
    fn e09_file_and_output_caps() {
        let rig = Rig::new();
        let p = rig.policy();
        let big = rig.work().join("big.bin");
        let o = run_ok(&rig, &p, &["bigfile", s(&big), "64000000"]);
        assert_eq!(o.ended, Ended::OutputLimit);
        let o = run_ok(&rig, &p, &["print", "8000000"]);
        assert_eq!(o.ended, Ended::OutputLimit);
        assert!(o.stdout.len() as u64 <= p.max_output_bytes);
        // A file over the cap left by an earlier run does not stop the next one.
        let o = run_ok(&rig, &p, &["exit", "0"]);
        assert_eq!(o.ended, Ended::Exited(0));
    }

    #[test]
    fn e10_no_process_outlives_the_run() {
        let rig = Rig::new();
        let pid_file = rig.work().join("pid.txt");
        let o = run_ok(&rig, &rig.policy(), &["detach", s(&pid_file)]);
        let out = String::from_utf8_lossy(&o.stdout);
        assert_eq!(o.ended, Ended::Exited(DID), "{out}");
        let pid: u32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let mut sys = sysinfo::System::new();
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        assert!(
            sys.process(sysinfo::Pid::from_u32(pid)).is_none(),
            "grandchild {pid} survived"
        );
    }

    #[test]
    fn e11_environment_is_not_inherited() {
        let rig = Rig::new();
        let o = run_ok(&rig, &rig.policy(), &["env", "PATH"]);
        assert_blocked(&o, "env PATH");
        let o = run_ok(&rig, &rig.policy(), &["env", "USERPROFILE"]);
        assert_blocked(&o, "env USERPROFILE");
    }

    #[test]
    fn e12_other_processes_cannot_be_opened() {
        let rig = Rig::new();
        let me = std::process::id().to_string();
        let o = run_ok(&rig, &rig.policy(), &["parent", &me]);
        assert_blocked(&o, "open parent");
    }

    #[test]
    fn e13_allowed_actions_still_work() {
        let rig = Rig::new();
        let ro = rig.outside("ro.txt");
        std::fs::write(&ro, b"readable").unwrap();
        let mut p = rig.policy();
        p.read_only.push(ro.clone());
        p.env.push(("THME_PROBE".into(), "1".into()));
        assert_eq!(
            run_ok(&rig, &p, &["read", s(&ro)]).ended,
            Ended::Exited(DID)
        );
        let w = rig.work().join("out.txt");
        assert_eq!(
            run_ok(&rig, &p, &["write", s(&w)]).ended,
            Ended::Exited(DID)
        );
        assert_eq!(std::fs::read(&w).unwrap(), b"probe");
        assert_eq!(
            run_ok(&rig, &p, &["env", "THME_PROBE"]).ended,
            Ended::Exited(DID)
        );
        assert_eq!(run_ok(&rig, &p, &["exit", "7"]).ended, Ended::Exited(7));
        let args: Vec<OsString> = vec!["sandbox".into(), "probe".into(), "echo-stdin".into()];
        let o = run(&p, &rig.probe, &args, b"hello").unwrap();
        assert_eq!(o.stdout, b"hello");
        // Grants are removed afterwards: the work folder no longer lists a container SID.
        let acl = std::process::Command::new("icacls")
            .arg(rig.work())
            .output()
            .unwrap();
        assert!(!String::from_utf8_lossy(&acl.stdout).contains("S-1-15-2"));
    }

    #[test]
    fn parallel_runs_do_not_disturb_each_other() {
        let rig = Rig::new();
        let p = rig.policy();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let (rig, p) = (&rig, &p);
                    scope.spawn(move || {
                        let w = rig.work().join(format!("p{i}.txt"));
                        run_ok(rig, p, &["write", s(&w)]).ended
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), Ended::Exited(DID));
            }
        });
        let acl = std::process::Command::new("icacls")
            .arg(&rig.probe)
            .output()
            .unwrap();
        assert!(!String::from_utf8_lossy(&acl.stdout).contains("S-1-15-2"));
    }

    #[test]
    fn invalid_policies_are_refused_before_starting() {
        let rig = Rig::new();
        let mut p = rig.policy();
        p.work_dir = rig.outside("missing");
        assert!(matches!(
            rig.probe(&p, &["exit", "0"]),
            Err(SandboxError::Policy(_))
        ));
        let mut p = rig.policy();
        p.env.push(("A=B".into(), "x".into()));
        assert!(matches!(
            rig.probe(&p, &["exit", "0"]),
            Err(SandboxError::Policy(_))
        ));
    }
}
