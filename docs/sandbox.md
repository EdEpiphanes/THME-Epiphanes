# Developer sandbox: design

Status: design done (v0.2.7.5). Implementation (v0.2.7.6, THME-3020): **Windows built**, with
every escape test passing (`tests/sandbox_escape.rs`); Linux and macOS refuse to run anything
until their backends land. This document is the contract each backend must meet.

## Why

Three kinds of code may reach THME that nobody has reviewed:

| Source | Example | Roadmap |
| :--- | :--- | :--- |
| Remote-code models | `auto_map` in `config.json`, run with `trust_remote_code` | v0.2.7.6 |
| Benchmark code | Unit tests for code tasks (pass@k), custom metrics that are more than data | v0.2.8 (dossier R.4, WP32) |
| Plugins | Out-of-process codecs, transforms and cost models (dossier L.4) | v0.3 |

`thme scan` finds such code but does not run it. The sandbox is where it runs when the user
chooses to run it. Code that the user did not choose to run is never started, sandbox or not.

## Threat model

The code inside is assumed hostile. It may try to:

1. Read files it was not given (keys, browser data, other models, `thme.key`).
2. Write or delete files outside its work folder, including through symbolic or hard links.
3. Reach the network: send data out, download a second stage, or talk to local services on
   `127.0.0.1`.
4. Exhaust the machine: memory, CPU time, processes (fork bomb), disk, output volume.
5. Outlive the run: leave child processes behind, or detach from the parent.
6. Read secrets from the environment (tokens in variables such as `HF_TOKEN`).
7. Signal, trace or inject into other processes, including THME itself.

Out of scope, stated so nobody relies on it:

- Kernel or hypervisor exploits and hardware side channels. The sandbox reduces the attack
  surface; it does not replace a virtual machine for code you expect to be targeted.
- GPU access. Version 1 gives the code no GPU.
- Denial of service against the work folder itself: the code may fill or corrupt its own folder,
  up to the disk cap.

## Interface

One module, `src/sandbox`, with a small interface. Callers (the remote-code runner, the benchmark
runner, plugin hosting) do not touch the per-OS mechanisms.

```rust
pub struct Policy {
    pub work_dir: PathBuf,          // read-write; the only writable place
    pub read_only: Vec<PathBuf>,    // e.g. the interpreter and its libraries
    pub env: Vec<(String, String)>, // the whole environment; nothing is inherited
    pub wall_time: Duration,
    pub cpu_time: Duration,
    pub memory_bytes: u64,
    pub max_processes: u32,
    pub max_file_bytes: u64,        // largest file the code may write
    pub max_output_bytes: u64,      // stdout and stderr, each
}

pub enum Ended { Exited(i32), Signalled(i32), TimeLimit, CpuLimit, MemoryLimit, OutputLimit, ProcessLimit }

pub struct Outcome { pub ended: Ended, pub stdout: Vec<u8>, pub stderr: Vec<u8>, pub wall: Duration }

pub fn run(policy: &Policy, program: &Path, args: &[OsString], stdin: &[u8])
    -> Result<Outcome, SandboxError>;

pub fn check() -> Support; // what this host can enforce, layer by layer
```

Rules of the interface:

- **Fail closed.** If any layer the policy needs cannot be enforced on this host, `run` returns
  `SandboxError::Unsupported { layer, reason }` and starts nothing. There is no degraded mode
  and no flag that turns layers off. `thme sandbox check` prints `Support` so a user can see why.
- **Network is always off.** `Policy` has no network field. Code that needs data gets it as files
  in `work_dir`.
- **Nothing is inherited.** No environment variables, no open file handles other than the three
  pipes, current directory set to `work_dir`.
- **Every limit is reported.** A run that hit a limit says which one; the caller decides what a
  limit means (a benchmark item scores as failed, a model load is refused).
- **One run, one tree.** When `run` returns, every process the code started is gone.

CLI, for developers: `thme sandbox check` and `thme sandbox run --work DIR [--read-only PATH]...
[--time S] [--memory MB] -- PROGRAM ARGS`.

## Layers per OS

Each threat is covered by at least one layer, and the escape tests prove it. Where a mechanism's
exact behaviour is not yet confirmed, the table says so and the escape test decides; if the test
fails, the layer counts as unsupported and `run` refuses.

### Linux

| Threat | Layer |
| :--- | :--- |
| Files (1, 2) | Landlock ruleset: read-only paths, read-write `work_dir`, nothing else. ABI 1 (Linux 5.13) minimum; link and rename rules need ABI 2 (5.19); refused below that |
| Network (3) | seccomp filter: `socket()` refused for every family, Unix sockets included (`socketpair()` still works for pipes between the program's own processes); `io_uring` refused, since its socket operations would bypass the filter. Landlock ABI 4 (6.7) TCP rules added on top where available. seccomp cannot read a socket address, so blocking `socket()` itself is what closes Unix and abstract sockets |
| Exhaustion (4) | `setrlimit`: `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NPROC`, `RLIMIT_FSIZE`, `RLIMIT_CORE = 0`; parent caps pipe reads |
| Outliving (5) | New process group and session; `PR_SET_PDEATHSIG`; parent kills the group on return. A cgroup is used when one is delegated to the user, otherwise the process group |
| Environment (6) | Empty environment plus `Policy::env` |
| Other processes (7) | `PR_SET_NO_NEW_PRIVS`; seccomp refuses `ptrace`, `process_vm_readv/writev`; Landlock ABI 6 signal scoping where available |

Namespaces (`unshare` of network and mount) would be a stronger network layer, but unprivileged
user namespaces are disabled on several distributions. They may be added as an extra layer; they
are never the only one.

### Windows

| Threat | Layer |
| :--- | :--- |
| Files (1, 2) | AppContainer: a per-run container profile with no capabilities. Access is granted by ACL to `work_dir` (read-write), the program and each read-only path; only the run's own entries are added and revoked, so parallel runs do not disturb each other. Windows also gives the container a private folder (`LOCALAPPDATA`, `TEMP` and `TMP` point there), deleted with the profile |
| Network (3) | AppContainer without network capabilities; loopback exemptions are never added. Escape test E4: TCP, UDP and DNS are blocked, outward and to loopback listeners |
| Exhaustion (4) | Job object: memory limit for the whole job, CPU time limit (user-mode time, so a run can use somewhat more wall time than `cpu_time` before it is stopped), active process limit; parent caps pipe reads. File size: no per-file cap in the OS, so the parent checks `work_dir` every 50 ms and stops the run when a file it wrote passes `max_file_bytes` |
| Outliving (5) | Job object with kill-on-job-close; the child is created suspended, assigned to the job, then resumed, so no process escapes the job |
| Environment (6) | Explicit environment block |
| Other processes (7) | AppContainer token (low integrity, no access to other processes' objects) |

### macOS

| Threat | Layer |
| :--- | :--- |
| Files (1, 2) | Seatbelt profile (`sandbox_init` or `/usr/bin/sandbox-exec`): `(deny default)`, read for listed paths, read-write for `work_dir` |
| Network (3) | Seatbelt `(deny network*)` |
| Exhaustion (4) | `setrlimit` for CPU, processes, file size. Memory: `RLIMIT_AS` is reported not to be enforced on macOS, so the parent samples the tree's resident size and kills it over the cap; the escape test measures how far over it can get |
| Outliving (5) | Process group; parent kills the group on return |
| Environment (6) | Empty environment plus `Policy::env` |
| Other processes (7) | Seatbelt denies `process-info*` and signals outside the sandbox |

Seatbelt is undocumented and marked deprecated, but it is what macOS offers; browsers and
developer tools use it. If Apple removes it, macOS becomes unsupported (fail closed), not
unprotected.

## Escape tests

The implementation ships with a probe program built from this repository (Rust, no interpreter
needed in CI). Each test runs the probe inside the sandbox and asserts the attempt fails and the
run reports the right `Ended`. All three CI operating systems run every test; a test that cannot
run on a host fails, it is not skipped.

| # | Attempt | Expected |
| :--- | :--- | :--- |
| E1 | Read a canary file outside `work_dir` (in the temp folder, readable by the user) | Permission denied |
| E2 | Write outside `work_dir`; create a symbolic link and a hard link in `work_dir` pointing out, then write through each | Denied, target unchanged |
| E3 | Rename a file out of `work_dir`; `..` in paths | Denied |
| E4 | TCP and UDP to `127.0.0.1` (a listener started by the test) and to a routable address; DNS lookup | Each fails; the listener sees no connection |
| E5 | Connect to a Unix socket and an abstract socket the test opened outside (Linux, macOS) | Denied |
| E6 | Allocate past `memory_bytes` | `MemoryLimit` |
| E7 | Spin past `cpu_time`; sleep past `wall_time` | `CpuLimit`, `TimeLimit` |
| E8 | Fork until refused | `ProcessLimit`, no process left |
| E9 | Write a file past `max_file_bytes`; print past `max_output_bytes` | Write fails or `OutputLimit` |
| E10 | Start a detached grandchild that outlives the probe | Gone when `run` returns |
| E11 | Read an environment variable set in the parent | Not present |
| E12 | Signal or trace the parent | Denied |
| E13 | Allowed things still work: read a read-only path, write in `work_dir`, exit code passed back | Succeeds |

E13 matters as much as the rest: a sandbox that blocks everything proves nothing.

## Alternatives considered

- **WebAssembly runtime (Wasmtime).** Strong isolation, but cannot run CPython with native
  extensions, which remote-code models and most benchmark code need, and it would roughly double
  the CLI binary against its 10 MB limit.
- **Containers (Docker, Podman).** Good isolation where installed, but an external runtime that
  many users do not have, and unavailable on stock Windows and macOS without a VM. May be offered
  later as an optional backend behind the same interface.
- **Virtual machine per run.** Strongest; too slow and heavy for running a benchmark's thousands
  of items. Recommended in the threat model for code expected to be targeted.

Chosen: the OS's own mechanisms, behind one interface, failing closed.

## Dependencies (proposal, needs owner approval before v0.2.7.6)

| Crate | OS | Use | Licence |
| :--- | :--- | :--- | :--- |
| `landlock` | Linux | Landlock rulesets with ABI fallback | MIT or Apache-2.0 |
| `seccompiler` | Linux | seccomp filters (from the rust-vmm project) | Apache-2.0 or BSD-3-Clause |
| `windows-sys` | Windows | AppContainer, job objects, ACLs | MIT or Apache-2.0 |
| `libc` | Linux, macOS | `setrlimit`, `prctl`, process groups (already in the tree indirectly) | MIT or Apache-2.0 |

macOS needs no new crate if the profile is applied through `/usr/bin/sandbox-exec`. All are pure
Rust bindings with no C runtime to ship.

## Where the sandbox is required

Once built, these paths start code only through `sandbox::run`, with no override:

- Running a model's remote code (`auto_map`), after `thme scan` has reported it.
- Benchmark code tasks and custom metrics (v0.2.8 runner).
- Plugins (v0.3).
- Code proposed by a teacher model in the autonomy loop (dossier Q), after review.
