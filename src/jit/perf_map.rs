//! `HERBERT_PERF_MAP=1`: the emitted code named for `perf`, in `/tmp/perf-<pid>.map`.
//!
//! One line per piece of code placed on an executable page, in perf's format (`START SIZE name`,
//! both in hex without `0x`): the entries and kernels of every generated program, the GEMM
//! micro-kernels, the attention, the element-wise, norm, recurrence and convolution kernels. Every
//! line is written and flushed when its page is mapped, so a server killed later leaves a whole
//! file. Nothing is written without the variable; a page released keeps its line (perf reads the
//! file once, at report time, and the addresses of a later page may repeat).
use std::io::Write;
use std::sync::{Mutex, OnceLock};

fn file() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| {
        if !std::env::var("HERBERT_PERF_MAP").is_ok_and(|v| !v.is_empty() && v != "0") {
            return None;
        }
        let path = format!("/tmp/perf-{}.map", std::process::id());
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => {
                eprintln!("[perf-map] emitted code named in {path}");
                Some(Mutex::new(f))
            }
            Err(e) => {
                eprintln!("[perf-map] cannot open {path}: {e}");
                None
            }
        }
    })
    .as_ref()
}

/// Is the map being written?
pub fn enabled() -> bool {
    file().is_some()
}

/// Name `len` bytes of code at `start`; `name` is only built when the map is written.
pub fn register(start: *const u8, len: usize, name: impl FnOnce() -> String) {
    let Some(f) = file() else { return };
    if len == 0 {
        return;
    }
    // perf splits the line at the first two spaces: the name keeps its own.
    let line = format!("{:x} {:x} herbert::{}\n", start as usize, len, name());
    let mut f = f.lock().unwrap_or_else(|e| e.into_inner());
    let _ = f.write_all(line.as_bytes());
    let _ = f.flush();
}

/// Name the pieces of one page laid out at `offsets` (each piece runs to the next offset, the last
/// to `total`).
pub fn register_pieces(
    base: *const u8,
    offsets: &[usize],
    lens: Option<&[usize]>,
    total: usize,
    name: impl Fn(usize) -> String,
) {
    if !enabled() {
        return;
    }
    for (i, &at) in offsets.iter().enumerate() {
        let end = lens
            .map(|l| at + l[i])
            .unwrap_or_else(|| offsets.get(i + 1).copied().unwrap_or(total));
        register(base.wrapping_add(at), end.saturating_sub(at), || name(i));
    }
}
