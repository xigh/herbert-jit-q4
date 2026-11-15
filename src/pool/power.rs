//! The process's standing with the operating system's power manager, set once
//! by `Engine::load` before anything is read or computed. Nothing here knows
//! a model or a kernel.

use std::sync::Once;

/// Takes the process out of the power throttling the system may apply to it.
///
/// On Windows 11, a process with no window of its own — the engine started by
/// a service, or through WMI (`Win32_Process.Create`), outside any
/// session — is judged background work and gets EcoQoS: its threads run on
/// the LPE cores. Measured on a Lunar Lake (Core Ultra 7 258V, 4 P + 4 LPE, on
/// mains), Qwen3.5-0.8B at 4 threads, started through WMI: 8.9 tokens/s at
/// decode (41 at prefill) as is, 107 (247) with execution-speed throttling
/// switched off for the process (`ProcessPowerThrottling`, `ControlMask` =
/// `EXECUTION_SPEED`, `StateMask` = 0), the threads then on the P cores — the
/// same as raising the priority class to High, without touching the priority.
/// Elsewhere, a no-op.
///
/// The first call does the work and logs one line; the others return at once.
pub fn opt_out_of_power_throttling() {
    static ONCE: Once = Once::new();
    ONCE.call_once(opt_out);
}

#[cfg(windows)]
fn opt_out() {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_STATE,
    };
    // The bit in ControlMask says "this policy is mine, not the system's"; the
    // same bit clear in StateMask says "off".
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: 0,
    };
    // SAFETY: the pseudo-handle of the current process needs no closing, and
    // `state` outlives the call, which only reads it.
    let ok = unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &state as *const PROCESS_POWER_THROTTLING_STATE as *const core::ffi::c_void,
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        )
    };
    if ok != 0 {
        eprintln!("[power] execution-speed throttling (EcoQoS) off for the process");
    } else {
        eprintln!(
            "[power] execution-speed throttling left to Windows, SetProcessInformation failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

#[cfg(not(windows))]
fn opt_out() {}
