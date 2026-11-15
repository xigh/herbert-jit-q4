//! `herbert-jit-llama-serve service install|uninstall|start|stop|status [--name NAME] [MODEL
//! [server options]]`: the server as a service of the system, started with the session (Linux,
//! macOS) or the machine (Windows), restarted when it fails. NAME defaults to `herbert-jit-q4`; one
//! service per name.
//!
//! `install` takes the server's own command line. A model of Hugging Face (`org/name[@revision]`)
//! is looked up in the cache of the user who installs (downloaded beforehand with `hf download`),
//! and the service gets the absolute path of its snapshot, as it gets this executable's: it depends
//! neither on its working directory nor on its account. Removing that model or revision from the
//! cache (`hf cache rm`, `hf cache prune`) breaks the service: `uninstall` it first.
//!
//! - Linux: a systemd user unit, `~/.config/systemd/user/NAME.service`, enabled and started
//!   (`systemctl --user`); the server's output goes to the journal (`journalctl --user -u NAME`). A
//!   user unit stops at the last logout unless `loginctl enable-linger` keeps the user's manager
//!   running.
//! - macOS: a LaunchAgent, `~/Library/LaunchAgents/NAME.plist`, loaded in the session (`launchctl
//!   bootstrap gui/UID`), restarted on failure, its output in `~/Library/Logs/NAME.log`.
//! - Windows: a service of the service manager (automatic start, restarted on failure,
//!   LocalSystem), which runs this executable as `service-run`; the output in
//!   `%ProgramData%\herbert-jit-q4\NAME.log`. Needs an elevated prompt.

use std::path::{Path, PathBuf};

pub const DEFAULT_NAME: &str = "herbert-jit-q4";
/// The server's options that name a folder.
const PATH_OPTIONS: [&str; 2] = ["--log-dir", "--prefix-cache"];
pub const USAGE: &str = "herbert-jit-llama-serve service install|uninstall|start|stop|status [--name NAME] [MODEL_DIR|org/name[@rev] [server options]]";

/// `args`: what follows `service`.
pub fn command(args: &[String]) -> Result<(), String> {
    let (verb, mut rest) = args
        .split_first()
        .ok_or("service: install, uninstall, start, stop or status")?;
    let mut name = DEFAULT_NAME.to_string();
    if rest.first().map(String::as_str) == Some("--name") {
        name = rest.get(1).ok_or("--name needs a value")?.clone();
        rest = &rest[2..];
    }
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
    {
        return Err(format!(
            "--name {name}: letters, digits, '-', '_' and '.' only"
        ));
    }
    match verb.as_str() {
        "install" => {
            let exe = absolute(&std::env::current_exe().map_err(|e| e.to_string())?)?;
            let mut server = rest.to_vec();
            if let Some(model) = server.first_mut().filter(|a| !a.starts_with("--")) {
                let alias = crate::hub::alias_for(model);
                *model = absolute(&crate::hub::model_dir(model)?)?
                    .to_string_lossy()
                    .into_owned();
                if let Some(alias) = alias.filter(|_| !rest.iter().any(|a| a == "--alias")) {
                    server.extend(["--alias".to_string(), alias]);
                }
            }
            // The folders of the options made absolute too: a service starts elsewhere (`/` under
            // launchd). They need not exist yet; the server creates them.
            for i in 1..server.len() {
                if PATH_OPTIONS.contains(&server[i - 1].as_str()) {
                    server[i] = std::path::absolute(&server[i])
                        .map_err(|e| format!("{}: {e}", server[i]))?
                        .to_string_lossy()
                        .into_owned();
                }
            }
            os::install(&name, &exe, &server)
        }
        _ if !rest.is_empty() => Err(format!("service {verb} takes only --name")),
        "uninstall" => os::uninstall(&name),
        "start" => os::start(&name),
        "stop" => os::stop(&name),
        "status" => os::status(&name),
        other => Err(format!(
            "service {other}: install, uninstall, start, stop or status"
        )),
    }
}

fn absolute(p: &Path) -> Result<PathBuf, String> {
    let p = std::fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()))?;
    // Windows: canonicalize gives `\\?\C:\…`, which some tools refuse in a command line.
    #[cfg(windows)]
    if let Some(s) = p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        if !s.starts_with("UNC\\") {
            return Ok(PathBuf::from(s));
        }
    }
    Ok(p)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn run(program: &str, args: &[&str]) -> Result<(), String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("{program}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

#[cfg(target_os = "linux")]
mod os {
    use super::{home, run};
    use std::path::{Path, PathBuf};

    fn unit_path(name: &str) -> PathBuf {
        let config = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home().join(".config"));
        config
            .join("systemd")
            .join("user")
            .join(format!("{name}.service"))
    }

    /// systemd's quoting: each word in double quotes, `\` and `"` escaped, `%` doubled, `$`
    /// doubled.
    fn quote(word: &str) -> String {
        format!(
            "\"{}\"",
            word.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('%', "%%")
                .replace('$', "$$")
        )
    }

    pub fn install(name: &str, exe: &Path, args: &[String]) -> Result<(), String> {
        let path = unit_path(name);
        let exec: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(args.iter().cloned())
            .map(|w| quote(&w))
            .collect();
        let unit = format!(
            "[Unit]\nDescription=herbert-jit-llama-serve ({name})\nAfter=network-online.target\n\n[Service]\nExecStart={}\nRestart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=default.target\n",
            exec.join(" ")
        );
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(&path, unit).map_err(|e| format!("{}: {e}", path.display()))?;
        run("systemctl", &["--user", "daemon-reload"])?;
        run(
            "systemctl",
            &["--user", "enable", "--now", &format!("{name}.service")],
        )?;
        println!(
            "{} written, enabled and started (journalctl --user -u {name} -f)",
            path.display()
        );
        Ok(())
    }

    pub fn uninstall(name: &str) -> Result<(), String> {
        let path = unit_path(name);
        let _ = run(
            "systemctl",
            &["--user", "disable", "--now", &format!("{name}.service")],
        );
        match std::fs::remove_file(&path) {
            Ok(()) => println!("{} removed", path.display()),
            Err(_) => println!("{} was not there", path.display()),
        }
        run("systemctl", &["--user", "daemon-reload"])
    }

    pub fn start(name: &str) -> Result<(), String> {
        run(
            "systemctl",
            &["--user", "start", &format!("{name}.service")],
        )
    }

    pub fn stop(name: &str) -> Result<(), String> {
        run("systemctl", &["--user", "stop", &format!("{name}.service")])
    }

    pub fn status(name: &str) -> Result<(), String> {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "status", "--no-pager", &format!("{name}.service")])
            .status();
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod os {
    use super::{home, run};
    use std::path::{Path, PathBuf};

    fn plist_path(name: &str) -> PathBuf {
        home()
            .join("Library")
            .join("LaunchAgents")
            .join(format!("{name}.plist"))
    }

    fn domain() -> String {
        format!("gui/{}", unsafe { libc::getuid() })
    }

    fn xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    pub fn install(name: &str, exe: &Path, args: &[String]) -> Result<(), String> {
        let path = plist_path(name);
        let log = home()
            .join("Library")
            .join("Logs")
            .join(format!("{name}.log"));
        let words: String = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(args.iter().cloned())
            .map(|w| format!("    <string>{}</string>\n", xml(&w)))
            .collect();
        let plist = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key><string>{}</string>\n  <key>ProgramArguments</key>\n  <array>\n{words}  </array>\n  <key>RunAtLoad</key><true/>\n  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>\n  <key>ProcessType</key><string>Interactive</string>\n  <key>StandardOutPath</key><string>{}</string>\n  <key>StandardErrorPath</key><string>{}</string>\n</dict>\n</plist>\n",
            xml(name),
            xml(&log.to_string_lossy()),
            xml(&log.to_string_lossy())
        );
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        let _ = std::fs::create_dir_all(log.parent().unwrap());
        std::fs::write(&path, plist).map_err(|e| format!("{}: {e}", path.display()))?;
        let _ = run(
            "/bin/launchctl",
            &["bootout", &format!("{}/{name}", domain())],
        );
        bootstrap(&path)?;
        println!(
            "{} written and loaded (log: {})",
            path.display(),
            log.display()
        );
        Ok(())
    }

    /// Right after a bootout, launchd sometimes still refuses ("Input/output error"): a few
    /// tries.
    fn bootstrap(path: &Path) -> Result<(), String> {
        let path = path.to_string_lossy();
        let mut last = String::new();
        for _ in 0..10 {
            match run("/bin/launchctl", &["bootstrap", &domain(), &path]) {
                Ok(()) => return Ok(()),
                Err(e) => last = e,
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        Err(last)
    }

    pub fn uninstall(name: &str) -> Result<(), String> {
        let _ = run(
            "/bin/launchctl",
            &["bootout", &format!("{}/{name}", domain())],
        );
        let path = plist_path(name);
        match std::fs::remove_file(&path) {
            Ok(()) => println!("{} removed", path.display()),
            Err(_) => println!("{} was not there", path.display()),
        }
        Ok(())
    }

    pub fn start(name: &str) -> Result<(), String> {
        bootstrap(&plist_path(name))
    }

    pub fn stop(name: &str) -> Result<(), String> {
        run(
            "/bin/launchctl",
            &["bootout", &format!("{}/{name}", domain())],
        )
    }

    pub fn status(name: &str) -> Result<(), String> {
        let _ = std::process::Command::new("/bin/launchctl")
            .args(["print", &format!("{}/{name}", domain())])
            .status();
        Ok(())
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
    use windows_service::service::{
        ServiceAccess, ServiceAction, ServiceActionType, ServiceErrorControl,
        ServiceFailureActions, ServiceFailureResetPeriod, ServiceInfo, ServiceStartType,
        ServiceState, ServiceType,
    };
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

    fn manager(access: ServiceManagerAccess) -> Result<ServiceManager, String> {
        ServiceManager::local_computer(None::<&str>, access)
            .map_err(|e| format!("service manager (an elevated prompt is needed): {e}"))
    }

    fn open(
        name: &str,
        access: ServiceAccess,
    ) -> Result<windows_service::service::Service, String> {
        manager(ServiceManagerAccess::CONNECT)?
            .open_service(name, access)
            .map_err(|e| format!("service {name}: {e}"))
    }

    pub fn log_path(name: &str) -> PathBuf {
        let data = std::env::var("ProgramData").unwrap_or_else(|_| r"C:\ProgramData".into());
        PathBuf::from(data)
            .join("herbert-jit-q4")
            .join(format!("{name}.log"))
    }

    pub fn install(name: &str, exe: &Path, args: &[String]) -> Result<(), String> {
        let mgr = manager(ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE)?;
        let launch: Vec<OsString> = ["service-run", "--name", name]
            .iter()
            .map(OsString::from)
            .chain(args.iter().map(OsString::from))
            .collect();
        let info = ServiceInfo {
            name: OsString::from(name),
            display_name: OsString::from(format!("herbert-jit-q4 ({name})")),
            service_type: ServiceType::OWN_PROCESS,
            start_type: ServiceStartType::AutoStart,
            error_control: ServiceErrorControl::Normal,
            executable_path: exe.to_path_buf(),
            launch_arguments: launch,
            dependencies: vec![],
            account_name: None,
            account_password: None,
        };
        let access = ServiceAccess::QUERY_STATUS
            | ServiceAccess::START
            | ServiceAccess::STOP
            | ServiceAccess::CHANGE_CONFIG
            | ServiceAccess::QUERY_CONFIG;
        let svc = match mgr.open_service(name, access) {
            Ok(svc) => {
                stop_and_wait(&svc)?;
                svc.change_config(&info).map_err(|e| e.to_string())?;
                svc
            }
            Err(_) => mgr
                .create_service(&info, access)
                .map_err(|e| format!("create service {name}: {e}"))?,
        };
        let _ = svc.set_description(
            "herbert-jit-llama-serve: llama-server's HTTP API on the CPU engine herbert-jit-q4",
        );
        let restart = |seconds| ServiceAction {
            action_type: ServiceActionType::Restart,
            delay: Duration::from_secs(seconds),
        };
        svc.update_failure_actions(ServiceFailureActions {
            reset_period: ServiceFailureResetPeriod::After(Duration::from_secs(24 * 3600)),
            reboot_msg: None,
            command: None,
            actions: Some(vec![restart(5), restart(10), restart(30)]),
        })
        .map_err(|e| e.to_string())?;
        let _ = svc.set_failure_actions_on_non_crash_failures(true);
        svc.start(&[] as &[&OsStr])
            .map_err(|e| format!("start {name}: {e}"))?;
        println!(
            "service {name} installed and started (log: {})",
            log_path(name).display()
        );
        Ok(())
    }

    pub fn uninstall(name: &str) -> Result<(), String> {
        let svc = match open(
            name,
            ServiceAccess::QUERY_STATUS | ServiceAccess::STOP | ServiceAccess::DELETE,
        ) {
            Ok(svc) => svc,
            Err(_) => {
                println!("service {name} was not there");
                return Ok(());
            }
        };
        stop_and_wait(&svc)?;
        svc.delete().map_err(|e| e.to_string())?;
        println!("service {name} removed");
        Ok(())
    }

    pub fn start(name: &str) -> Result<(), String> {
        open(name, ServiceAccess::START)?
            .start(&[] as &[&OsStr])
            .map_err(|e| e.to_string())
    }

    pub fn stop(name: &str) -> Result<(), String> {
        stop_and_wait(&open(
            name,
            ServiceAccess::QUERY_STATUS | ServiceAccess::STOP,
        )?)
    }

    pub fn status(name: &str) -> Result<(), String> {
        let s = open(name, ServiceAccess::QUERY_STATUS)?
            .query_status()
            .map_err(|e| e.to_string())?;
        println!(
            "service {name}: {:?}, pid {:?} (log: {})",
            s.current_state,
            s.process_id,
            log_path(name).display()
        );
        Ok(())
    }

    fn stop_and_wait(svc: &windows_service::service::Service) -> Result<(), String> {
        if svc.query_status().map_err(|e| e.to_string())?.current_state == ServiceState::Stopped {
            return Ok(());
        }
        let _ = svc.stop();
        let start = Instant::now();
        while svc.query_status().map_err(|e| e.to_string())?.current_state != ServiceState::Stopped
        {
            if start.elapsed() > Duration::from_secs(30) {
                return Err("the service does not stop".into());
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        Ok(())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
mod os {
    use std::path::Path;
    const NONE: &str = "no service manager known on this system";
    pub fn install(_: &str, _: &Path, _: &[String]) -> Result<(), String> {
        Err(NONE.into())
    }
    pub fn uninstall(_: &str) -> Result<(), String> {
        Err(NONE.into())
    }
    pub fn start(_: &str) -> Result<(), String> {
        Err(NONE.into())
    }
    pub fn stop(_: &str) -> Result<(), String> {
        Err(NONE.into())
    }
    pub fn status(_: &str) -> Result<(), String> {
        Err(NONE.into())
    }
}

/// Windows: the process the service manager starts (`service-run --name NAME server-args…`). It
/// says Running at once (the model loads after, as on the command line), sends its stderr to the
/// log, and runs `serve` on the server's arguments; a stop from the manager ends the process.
#[cfg(windows)]
pub fn run_service(args: &[String], serve: fn(Vec<String>)) -> Result<(), String> {
    use std::ffi::OsString;
    use std::time::Duration;
    use windows_service::service::{
        ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus,
        ServiceType,
    };
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
    use windows_service::{define_windows_service, service_dispatcher};

    static JOB: std::sync::OnceLock<(String, Vec<String>, fn(Vec<String>))> =
        std::sync::OnceLock::new();
    let (name, rest) = match args {
        [flag, name, rest @ ..] if flag == "--name" => (name.clone(), rest.to_vec()),
        _ => (DEFAULT_NAME.to_string(), args.to_vec()),
    };
    let _ = JOB.set((name.clone(), rest, serve));

    define_windows_service!(ffi_service_main, service_main);
    fn service_main(_: Vec<OsString>) {
        let (name, args, serve) = JOB.get().unwrap().clone();
        let status = std::sync::Arc::new(std::sync::OnceLock::new());
        let in_handler = status.clone();
        let handler = move |control| match control {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                if let Some(h) = in_handler.get() {
                    let _ = set(h, ServiceState::Stopped, ServiceControlAccept::empty());
                }
                std::thread::spawn(|| {
                    std::thread::sleep(Duration::from_millis(200));
                    std::process::exit(0)
                });
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        };
        let Ok(handle) = service_control_handler::register(&name, handler) else {
            return;
        };
        let _ = status.set(handle);
        let _ = set(
            &handle,
            ServiceState::Running,
            ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        );
        redirect_stderr(&os::log_path(&name));
        eprintln!("[service] {name}: {}", args.join(" "));
        serve(args);
        let _ = set(
            &handle,
            ServiceState::Stopped,
            ServiceControlAccept::empty(),
        );
    }
    fn set(
        h: &service_control_handler::ServiceStatusHandle,
        state: ServiceState,
        accept: ServiceControlAccept,
    ) -> windows_service::Result<()> {
        h.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: state,
            controls_accepted: accept,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(10),
            process_id: None,
        })
    }
    /// The service has no console: stderr (the server's log) appended to the file.
    fn redirect_stderr(path: &Path) {
        use std::os::windows::io::IntoRawHandle;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let handle = file.into_raw_handle();
            unsafe {
                windows_sys::Win32::System::Console::SetStdHandle(
                    windows_sys::Win32::System::Console::STD_ERROR_HANDLE,
                    handle as _,
                );
                windows_sys::Win32::System::Console::SetStdHandle(
                    windows_sys::Win32::System::Console::STD_OUTPUT_HANDLE,
                    handle as _,
                );
            }
        }
    }

    service_dispatcher::start(&name, ffi_service_main)
        .map_err(|e| format!("service dispatcher (run by the service manager only): {e}"))
}
