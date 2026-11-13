// SPDX-License-Identifier: AGPL-3.0-only
//! Windows backend: an AppContainer with no capabilities (no network, files only by explicit
//! grant) inside a job object (memory, CPU time, process count, kill on close, no UI access).
//! The child is created suspended and assigned to the job before it runs a single instruction.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::Write;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{FromRawHandle, OwnedHandle};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{
    CloseHandle, LocalFree, SetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT,
    INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
};
use windows_sys::Win32::Security::Authorization::{
    GetNamedSecurityInfoW, SetEntriesInAclW, SetNamedSecurityInfoW, EXPLICIT_ACCESS_W,
    GRANT_ACCESS, REVOKE_ACCESS, SE_FILE_OBJECT, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::Isolation::{
    CreateAppContainerProfile, DeleteAppContainerProfile, DeriveAppContainerSidFromAppContainerName,
};
use windows_sys::Win32::Security::{
    FreeSid, ACL, DACL_SECURITY_INFORMATION, NO_INHERITANCE, PSECURITY_DESCRIPTOR, PSID,
    SECURITY_ATTRIBUTES, SECURITY_CAPABILITIES, SUB_CONTAINERS_AND_OBJECTS_INHERIT,
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_ALL_ACCESS, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectAssociateCompletionPortInformation,
    JobObjectBasicUIRestrictions, JobObjectExtendedLimitInformation, SetInformationJobObject,
    TerminateJobObject, JOBOBJECT_ASSOCIATE_COMPLETION_PORT, JOBOBJECT_BASIC_UI_RESTRICTIONS,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_JOB_TIME, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::SystemServices::{
    JOB_OBJECT_MSG_ACTIVE_PROCESS_LIMIT, JOB_OBJECT_MSG_END_OF_JOB_TIME,
    JOB_OBJECT_MSG_JOB_MEMORY_LIMIT, JOB_OBJECT_UILIMIT_ALL,
};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess,
    InitializeProcThreadAttributeList, ResumeThread, TerminateProcess, UpdateProcThreadAttribute,
    WaitForSingleObject, CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
    EXTENDED_STARTUPINFO_PRESENT, LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION,
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
    STARTF_USESTDHANDLES, STARTUPINFOEXW,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatus, OVERLAPPED,
};

use super::{drain, file_sizes, wrote_over, Ended, Layer, Outcome, Policy, SandboxError, Support};

/// `HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)`.
const HR_ALREADY_EXISTS: i32 = 0x800700B7_u32 as i32;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn os_err(what: &str) -> SandboxError {
    SandboxError::Os(format!(
        "{what} failed: {}",
        std::io::Error::last_os_error()
    ))
}

fn wide(s: &OsStr) -> Vec<u16> {
    s.encode_wide().chain(std::iter::once(0)).collect()
}

/// Runs a closure when dropped; keeps cleanup on every return path.
struct OnDrop<F: FnMut()>(F);
impl<F: FnMut()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        (self.0)()
    }
}

/// A per-run AppContainer profile, deleted on drop.
struct Container {
    name: Vec<u16>,
    sid: PSID,
}

impl Container {
    fn create() -> Result<Self, SandboxError> {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let name = wide(OsStr::new(&format!(
            "thme.sandbox.{}.{n}",
            std::process::id()
        )));
        let mut sid: PSID = null_mut();
        // SAFETY: valid NUL-terminated strings; `sid` receives a SID freed in Drop.
        let mut hr = unsafe {
            CreateAppContainerProfile(
                name.as_ptr(),
                name.as_ptr(),
                name.as_ptr(),
                null(),
                0,
                &mut sid,
            )
        };
        if hr == HR_ALREADY_EXISTS {
            // SAFETY: as above.
            hr = unsafe { DeriveAppContainerSidFromAppContainerName(name.as_ptr(), &mut sid) };
        }
        if hr < 0 || sid.is_null() {
            return Err(SandboxError::Unsupported {
                layer: "files",
                reason: format!("AppContainer profile could not be created (HRESULT {hr:#010x})"),
            });
        }
        Ok(Self { name, sid })
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        // SAFETY: `sid` came from the profile calls; the name is NUL-terminated.
        unsafe {
            FreeSid(self.sid);
            DeleteAppContainerProfile(self.name.as_ptr());
        }
    }
}

/// Serialises ACL edits within this process, so parallel runs never undo each other's grants.
static ACL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Adds (`GRANT_ACCESS`) or removes (`REVOKE_ACCESS`, every entry for the SID) access for `sid`
/// on `path`, keeping every other entry as it is now.
fn edit_acl(path: &[u16], sid: PSID, mode: i32, access: u32, inherit: u32) -> u32 {
    let _lock = ACL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut dacl: *mut ACL = null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
    // SAFETY: valid out-pointers; `descriptor` is freed below.
    let rc = unsafe {
        GetNamedSecurityInfoW(
            path.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut descriptor,
        )
    };
    if rc != 0 {
        return rc;
    }
    let entry = EXPLICIT_ACCESS_W {
        grfAccessPermissions: access,
        grfAccessMode: mode,
        grfInheritance: inherit,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: null_mut(),
            MultipleTrusteeOperation: 0,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_UNKNOWN,
            ptstrName: sid as *mut u16,
        },
    };
    let mut new_dacl: *mut ACL = null_mut();
    // SAFETY: one valid entry over the current DACL; both allocations are freed below.
    let mut rc = unsafe { SetEntriesInAclW(1, &entry, dacl, &mut new_dacl) };
    if rc == 0 {
        // SAFETY: valid path and DACL.
        rc = unsafe {
            SetNamedSecurityInfoW(
                path.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                new_dacl,
                null(),
            )
        };
        unsafe { LocalFree(new_dacl as _) };
    }
    unsafe { LocalFree(descriptor as _) };
    rc
}

/// Access for the container on one path, revoked on drop. Only this container's entries are
/// added and removed, so concurrent runs on the same paths do not disturb each other.
struct Grant {
    path: Vec<u16>,
    sid: PSID,
}

impl Grant {
    fn new(path: &Path, sid: PSID, access: u32) -> Result<Self, SandboxError> {
        let inherit = if path.is_dir() {
            SUB_CONTAINERS_AND_OBJECTS_INHERIT
        } else {
            NO_INHERITANCE
        };
        let wpath = wide(path.as_os_str());
        let rc = edit_acl(&wpath, sid, GRANT_ACCESS, access, inherit);
        if rc != 0 {
            return Err(SandboxError::Os(format!(
                "granting access to {path:?} failed ({rc})"
            )));
        }
        Ok(Self { path: wpath, sid })
    }
}

impl Drop for Grant {
    fn drop(&mut self) {
        edit_acl(&self.path, self.sid, REVOKE_ACCESS, 0, NO_INHERITANCE);
    }
}

/// A job object with the policy's limits and a completion port for limit messages.
struct Job {
    job: HANDLE,
    port: HANDLE,
}

impl Job {
    fn create(policy: &Policy) -> Result<Self, SandboxError> {
        // SAFETY: plain creation calls; handles are closed in Drop.
        let job = unsafe { CreateJobObjectW(null(), null()) };
        if job.is_null() {
            return Err(os_err("CreateJobObjectW"));
        }
        let port = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, null_mut(), 0, 1) };
        let this = Self { job, port };
        if port.is_null() {
            return Err(os_err("CreateIoCompletionPort"));
        }
        let assoc = JOBOBJECT_ASSOCIATE_COMPLETION_PORT {
            CompletionKey: null_mut(),
            CompletionPort: port,
        };
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        let basic = &mut limits.BasicLimitInformation;
        basic.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_JOB_MEMORY
            | JOB_OBJECT_LIMIT_JOB_TIME
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        basic.PerJobUserTimeLimit = (policy.cpu_time.as_nanos() / 100).min(i64::MAX as u128) as i64;
        basic.ActiveProcessLimit = policy.max_processes;
        limits.JobMemoryLimit = usize::try_from(policy.memory_bytes).unwrap_or(usize::MAX);
        let ui = JOBOBJECT_BASIC_UI_RESTRICTIONS {
            UIRestrictionsClass: JOB_OBJECT_UILIMIT_ALL,
        };
        // SAFETY: each struct matches its information class and size.
        let ok = unsafe {
            SetInformationJobObject(
                job,
                JobObjectAssociateCompletionPortInformation,
                &assoc as *const _ as _,
                std::mem::size_of_val(&assoc) as u32,
            ) != 0
                && SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &limits as *const _ as _,
                    std::mem::size_of_val(&limits) as u32,
                ) != 0
                && SetInformationJobObject(
                    job,
                    JobObjectBasicUIRestrictions,
                    &ui as *const _ as _,
                    std::mem::size_of_val(&ui) as u32,
                ) != 0
        };
        if !ok {
            return Err(os_err("SetInformationJobObject"));
        }
        Ok(this)
    }

    /// Limit messages posted since the last call: (memory, cpu, processes).
    fn limits_hit(&self, wait_ms: u32) -> (bool, bool, bool) {
        let (mut mem, mut cpu, mut procs) = (false, false, false);
        let mut wait = wait_ms;
        loop {
            let mut msg = 0u32;
            let mut key = 0usize;
            let mut ov: *mut OVERLAPPED = null_mut();
            // SAFETY: valid out-pointers; the port is owned by self.
            let got =
                unsafe { GetQueuedCompletionStatus(self.port, &mut msg, &mut key, &mut ov, wait) };
            if got == 0 {
                return (mem, cpu, procs);
            }
            wait = 0;
            match msg {
                JOB_OBJECT_MSG_JOB_MEMORY_LIMIT => mem = true,
                JOB_OBJECT_MSG_END_OF_JOB_TIME => cpu = true,
                JOB_OBJECT_MSG_ACTIVE_PROCESS_LIMIT => procs = true,
                _ => {}
            }
        }
    }

    fn kill(&self) {
        // SAFETY: owned job handle.
        unsafe { TerminateJobObject(self.job, 1) };
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: owned handles; closing the job kills anything left (kill-on-close).
        unsafe {
            TerminateJobObject(self.job, 1);
            CloseHandle(self.job);
            if !self.port.is_null() {
                CloseHandle(self.port);
            }
        }
    }
}

/// An anonymous pipe: (parent end, child end). Only the child end is inheritable.
fn pipe(child_reads: bool) -> Result<(File, OwnedHandle), SandboxError> {
    let sa = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: null_mut(),
        bInheritHandle: 1,
    };
    let (mut r, mut w): (HANDLE, HANDLE) = (null_mut(), null_mut());
    // SAFETY: valid out-pointers; both handles are wrapped in owners right away.
    if unsafe { CreatePipe(&mut r, &mut w, &sa, 0) } == 0 {
        return Err(os_err("CreatePipe"));
    }
    let (r, w) = unsafe {
        (
            OwnedHandle::from_raw_handle(r),
            OwnedHandle::from_raw_handle(w),
        )
    };
    let (parent, child) = if child_reads { (w, r) } else { (r, w) };
    use std::os::windows::io::AsRawHandle;
    // SAFETY: the parent end must not be inherited.
    unsafe { SetHandleInformation(parent.as_raw_handle(), HANDLE_FLAG_INHERIT, 0) };
    Ok((File::from(parent), child))
}

/// Quotes one argument for the Windows command line (CommandLineToArgvW rules).
fn quote(arg: &OsStr, out: &mut Vec<u16>) {
    let a: Vec<u16> = arg.encode_wide().collect();
    let needs = a.is_empty()
        || a.iter()
            .any(|&c| c == b' ' as u16 || c == b'\t' as u16 || c == b'"' as u16);
    if !needs {
        out.extend(a);
        return;
    }
    out.push(b'"' as u16);
    let mut backslashes = 0;
    for &c in &a {
        if c == b'\\' as u16 {
            backslashes += 1;
        } else {
            if c == b'"' as u16 {
                out.extend(std::iter::repeat_n(b'\\' as u16, backslashes + 1));
            }
            backslashes = 0;
        }
        out.push(c);
    }
    out.extend(std::iter::repeat_n(b'\\' as u16, backslashes));
    out.push(b'"' as u16);
}

fn command_line(program: &Path, args: &[OsString]) -> Vec<u16> {
    let mut line = Vec::new();
    quote(program.as_os_str(), &mut line);
    for a in args {
        line.push(b' ' as u16);
        quote(a, &mut line);
    }
    line.push(0);
    line
}

fn env_block(policy: &Policy) -> Vec<u16> {
    let mut vars: Vec<(OsString, OsString)> = policy
        .env
        .iter()
        .map(|(k, v)| (OsString::from(k), OsString::from(v)))
        .collect();
    // System libraries need SystemRoot. CreateProcessW needs LOCALAPPDATA to place the
    // container's private folder, and rewrites LOCALAPPDATA, TEMP and TMP to that folder, so the
    // child never sees the caller's values. The folder is deleted with the profile.
    for name in ["SystemRoot", "LOCALAPPDATA"] {
        vars.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
        if let Some(value) = std::env::var_os(name) {
            vars.push((name.into(), value));
        }
    }
    vars.sort_by_key(|(k, _)| k.to_ascii_uppercase());
    let mut block = Vec::new();
    for (k, v) in vars {
        block.extend(k.encode_wide());
        block.push(b'=' as u16);
        block.extend(v.encode_wide());
        block.push(0);
    }
    block.push(0);
    block
}

pub(super) fn check() -> Support {
    let container = Container::create();
    let job = Job::create(&Policy::new(std::env::temp_dir()));
    let (c_ok, c_note) = match &container {
        Ok(_) => (true, "per-run AppContainer, no capabilities".to_string()),
        Err(e) => (false, e.to_string()),
    };
    let (j_ok, j_note) = match &job {
        Ok(_) => (true, "job object".to_string()),
        Err(e) => (false, e.to_string()),
    };
    let layer = |name, mechanism, ok: bool, note: &str| Layer {
        name,
        mechanism,
        supported: ok,
        note: note.to_string(),
    };
    Support {
        os: "windows",
        layers: vec![
            layer("files", "AppContainer + ACL grants", c_ok, &c_note),
            layer(
                "network",
                "AppContainer without network capabilities",
                c_ok,
                &c_note,
            ),
            layer(
                "limits",
                "job object: memory, CPU time, processes",
                j_ok,
                &j_note,
            ),
            layer(
                "file size",
                "parent watches the work folder",
                true,
                "polled every 50 ms",
            ),
            layer("lifetime", "job object, kill on close", j_ok, &j_note),
            layer(
                "environment",
                "explicit environment block",
                true,
                "nothing inherited",
            ),
            layer("other processes", "AppContainer token", c_ok, &c_note),
        ],
    }
}

pub(super) fn run(
    policy: &Policy,
    program: &Path,
    args: &[OsString],
    stdin: &[u8],
) -> Result<Outcome, SandboxError> {
    let before = file_sizes(&policy.work_dir);
    let container = Container::create()?;
    let job = Job::create(policy)?;
    let read = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE;
    let mut grants = vec![Grant::new(
        &policy.work_dir,
        container.sid,
        FILE_ALL_ACCESS,
    )?];
    grants.push(Grant::new(program, container.sid, read)?);
    for p in &policy.read_only {
        grants.push(Grant::new(p, container.sid, read)?);
    }

    let (mut stdin_parent, stdin_child) = pipe(true)?;
    let (stdout_parent, stdout_child) = pipe(false)?;
    let (stderr_parent, stderr_child) = pipe(false)?;

    use std::os::windows::io::AsRawHandle;
    let inherit: [HANDLE; 3] = [
        stdin_child.as_raw_handle(),
        stdout_child.as_raw_handle(),
        stderr_child.as_raw_handle(),
    ];
    let caps = SECURITY_CAPABILITIES {
        AppContainerSid: container.sid,
        Capabilities: null_mut(),
        CapabilityCount: 0,
        Reserved: 0,
    };
    let mut size = 0usize;
    // SAFETY: the first call only reports the size needed.
    unsafe { InitializeProcThreadAttributeList(null_mut(), 2, 0, &mut size) };
    let mut attr_buf = vec![0u64; size.div_ceil(8)];
    let attrs = attr_buf.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST;
    // SAFETY: the buffer is at least `size` bytes and 8-aligned.
    if unsafe { InitializeProcThreadAttributeList(attrs, 2, 0, &mut size) } == 0 {
        return Err(os_err("InitializeProcThreadAttributeList"));
    }
    let _attrs_guard = OnDrop(|| unsafe { DeleteProcThreadAttributeList(attrs) });
    // SAFETY: `caps` and `inherit` outlive CreateProcessW below.
    let ok = unsafe {
        UpdateProcThreadAttribute(
            attrs,
            0,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES as usize,
            &caps as *const _ as _,
            std::mem::size_of_val(&caps),
            null_mut(),
            null(),
        ) != 0
            && UpdateProcThreadAttribute(
                attrs,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                inherit.as_ptr() as _,
                std::mem::size_of_val(&inherit),
                null_mut(),
                null(),
            ) != 0
    };
    if !ok {
        return Err(os_err("UpdateProcThreadAttribute"));
    }

    let mut si: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
    si.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = inherit[0];
    si.StartupInfo.hStdOutput = inherit[1];
    si.StartupInfo.hStdError = inherit[2];
    si.lpAttributeList = attrs;

    let app = wide(program.as_os_str());
    let mut line = command_line(program, args);
    let env = env_block(policy);
    let cwd = wide(policy.work_dir.as_os_str());
    let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let started = Instant::now();
    // SAFETY: every pointer is valid for the call; the child starts suspended.
    let created = unsafe {
        CreateProcessW(
            app.as_ptr(),
            line.as_mut_ptr(),
            null(),
            null(),
            1,
            EXTENDED_STARTUPINFO_PRESENT
                | CREATE_SUSPENDED
                | CREATE_UNICODE_ENVIRONMENT
                | CREATE_NO_WINDOW,
            env.as_ptr() as _,
            cwd.as_ptr(),
            &si.StartupInfo,
            &mut pi,
        )
    };
    if created == 0 {
        return Err(os_err("CreateProcessW"));
    }
    // SAFETY: CreateProcessW returned owned handles.
    let process = unsafe { OwnedHandle::from_raw_handle(pi.hProcess) };
    let thread = unsafe { OwnedHandle::from_raw_handle(pi.hThread) };
    // SAFETY: owned handles.
    if unsafe { AssignProcessToJobObject(job.job, process.as_raw_handle()) } == 0 {
        let err = os_err("AssignProcessToJobObject");
        unsafe { TerminateProcess(process.as_raw_handle(), 1) };
        return Err(err);
    }
    drop((stdin_child, stdout_child, stderr_child));
    // SAFETY: owned thread handle of a suspended process now inside the job.
    if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX {
        return Err(os_err("ResumeThread"));
    }

    let over = Arc::new(AtomicBool::new(false));
    let cap = policy.max_output_bytes;
    let (o1, o2) = (over.clone(), over.clone());
    let out_thread = std::thread::spawn(move || drain(stdout_parent, cap, &o1));
    let err_thread = std::thread::spawn(move || drain(stderr_parent, cap, &o2));
    let input = stdin.to_vec();
    let in_thread = std::thread::spawn(move || {
        let _ = stdin_parent.write_all(&input);
    });

    let mut ended = None;
    let mut last_poll = Instant::now();
    loop {
        // SAFETY: owned process handle.
        let done = unsafe { WaitForSingleObject(process.as_raw_handle(), 20) } == WAIT_OBJECT_0;
        let (mem, cpu, procs) = job.limits_hit(0);
        let reason = if mem {
            Some(Ended::MemoryLimit)
        } else if procs {
            Some(Ended::ProcessLimit)
        } else if cpu {
            Some(Ended::CpuLimit)
        } else if over.load(Ordering::SeqCst)
            || (last_poll.elapsed() >= Duration::from_millis(50)
                && wrote_over(&policy.work_dir, &before, policy.max_file_bytes))
        {
            Some(Ended::OutputLimit)
        } else if started.elapsed() >= policy.wall_time {
            Some(Ended::TimeLimit)
        } else {
            None
        };
        if last_poll.elapsed() >= Duration::from_millis(50) {
            last_poll = Instant::now();
        }
        if reason.is_some() {
            ended = reason;
            job.kill();
            break;
        }
        if done {
            break;
        }
    }
    let wall = started.elapsed();
    // A limit message can trail the exit by a moment.
    if ended.is_none() {
        let (mem, cpu, procs) = job.limits_hit(10);
        ended = if mem {
            Some(Ended::MemoryLimit)
        } else if procs {
            Some(Ended::ProcessLimit)
        } else if cpu {
            Some(Ended::CpuLimit)
        } else if wrote_over(&policy.work_dir, &before, policy.max_file_bytes) {
            Some(Ended::OutputLimit)
        } else {
            None
        };
    }
    job.kill();
    // SAFETY: owned process handle.
    unsafe { WaitForSingleObject(process.as_raw_handle(), 5_000) };
    let mut code = 0u32;
    unsafe { GetExitCodeProcess(process.as_raw_handle(), &mut code) };
    let stdout = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();
    let _ = in_thread.join();
    drop(grants);
    drop(job);
    drop(container);
    let ended = ended.unwrap_or(if over.load(Ordering::SeqCst) {
        Ended::OutputLimit
    } else {
        Ended::Exited(code as i32)
    });
    Ok(Outcome {
        ended,
        stdout,
        stderr,
        wall,
    })
}
