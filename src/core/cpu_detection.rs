//! How many workers by default: the performance cores of the machine.

/// Number of performance cores.
///
/// On Linux, the physical P cores of a hybrid (`/sys/devices/cpu_core/cpus`),
/// otherwise the physical cores, SMT siblings excluded — one full L2 per
/// thread. On Windows, the physical cores of the highest efficiency class
/// (`GetLogicalProcessorInformationEx`): the P cores of a hybrid, every
/// physical core elsewhere — on a Lunar Lake (4 P + 4 LPE) 4 workers decode
/// Qwen3.5-0.8B at 107 tokens/s, all 8 at 80. On an Apple machine with few performance cores,
/// the performance cores and half of the efficiency cores: on the M5 (4 + 6),
/// 7 workers decode at 14.1 tokens/s where 4 give 9.7 and all 10 give 9.4 —
/// the spinning barriers then wait for the cores the system takes back
/// (tasks handed out by ticket, `Knobs::task_grain`). With eight performance
/// cores or more, the performance cores alone: on the M3 Ultra (20 + 8) the
/// CPU's share of the memory saturates at 256 GB/s from 8 threads, 20 decode
/// at 26.5 tokens/s, 16 at 27.0, 8 at 27.6, and 24 (P + E/2) fall to 4–13 —
/// the efficiency cores of the second die hold every barrier.
///
/// Cached after the first call.
pub fn get_performance_cpu_count() -> usize {
    use std::sync::OnceLock;
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            // Hybrid Intel (P + E, `/sys/devices/cpu_core/cpus`): only the P
            // cores — measured on a Lunar Lake, 3 P cores alone decode faster
            // than 8 threads split equally. The pool pins P cores first
            // (`pool::topology`), so "as many workers as P cores" = the P cores.
            let p = count_physical_p_cores_linux();
            let count = if p > 0 {
                p
            } else {
                match count_physical_cores_linux() {
                    0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
                    n => n,
                }
            };
            // A container's cpuset (or `taskset`) leaves fewer CPUs than the machine has: no more
            // workers than CPUs allowed.
            allowed_cpus().map_or(count, |a| count.min(a.len().max(1)))
        }

        #[cfg(target_os = "macos")]
        {
            use crate::core::isa::cores_of_level;
            if let Some(p) = cores_of_level(0) {
                return if p >= 8 {
                    p
                } else {
                    p + cores_of_level(1).unwrap_or(0) / 2
                };
            }
            std::thread::available_parallelism().map_or(1, |n| n.get())
        }

        #[cfg(windows)]
        {
            let p = count_physical_top_class_cores_windows();
            if p > 0 {
                return p;
            }
            std::thread::available_parallelism().map_or(1, |n| n.get())
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        }
    })
}

/// The physical cores of the highest `EfficiencyClass` (one
/// `RelationProcessorCore` record per physical core, SMT siblings included in
/// it). A machine that is not hybrid reports one class for every core, so all
/// of them count. `0` when the call fails.
#[cfg(windows)]
fn count_physical_top_class_cores_windows() -> usize {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore, PROCESSOR_RELATIONSHIP,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut len = 0u32;
    // SAFETY: a null buffer asks for the length only.
    unsafe {
        GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len)
    };
    if len == 0 {
        return 0;
    }
    // u64 words: the records need the alignment of their structure.
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: `buf` holds at least `len` bytes, aligned for the records.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut len,
        )
    };
    if ok == 0 {
        return 0;
    }
    let bytes = buf.as_ptr() as *const u8;
    let class_at = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous)
        + std::mem::offset_of!(PROCESSOR_RELATIONSHIP, EfficiencyClass);
    let mut classes = Vec::new();
    // The records are variable-sized and shorter than the structure, whose union is sized for the
    // largest relationship: walk by `Size`, reading the two header words and the class byte
    // only (bounds checked against `len`, not against the structure's size, which would drop
    // the last core).
    let header = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous);
    let (relationship_at, size_at) = (
        std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship),
        std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size),
    );
    let mut at = 0usize;
    while at + header <= len as usize {
        // SAFETY: `at + header` is within the `len` bytes the call wrote, `at` on a record
        // boundary, and `buf` aligned for the record's 4-byte header words.
        let (relationship, size) = unsafe {
            (
                *(bytes.add(at + relationship_at) as *const i32),
                *(bytes.add(at + size_at) as *const u32) as usize,
            )
        };
        if size == 0 || at + size > len as usize {
            break;
        }
        if relationship == RelationProcessorCore && class_at < size {
            // SAFETY: `class_at < size`, inside this record.
            classes.push(unsafe { *bytes.add(at + class_at) });
        }
        at += size;
    }
    let top = classes.iter().copied().max().unwrap_or(0);
    classes.iter().filter(|&&c| c == top).count()
}

/// Participants of the decode by default, when not the whole pool.
///
/// The decode reads the weights once per token and is bound by the memory
/// the CPU can draw; the prefill wants every performance core. On a Mac with
/// more than sixteen performance cores the decode takes sixteen of the pool
/// (`Tuning::decode_workers`) — measured on the M3 Ultra, pool of 20, three
/// interleaved repetitions: 27.37 tokens/s at 20, 27.57 at 8, 27.53
/// at 12, **27.69 at 16** (26.56 → 27.09 after a 1 666-token prompt), the
/// prefill unchanged (244 tokens/s). Elsewhere, `None`: the whole pool.
pub fn default_decode_workers() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        if let Some(p) = crate::core::isa::cores_of_level(0) {
            return decode_workers_for(p);
        }
    }
    None
}

/// The prefill barriers by default: on a Mac, spin then give the core back
/// (`Executor::SpmdHybrid` with `HybridWait::Yield`, `hybrid_spin` pauses
/// first). With every performance core in the team, a participant that a
/// third-party thread preempts during its busy wait otherwise loses up to a
/// scheduler quantum (~10 ms) at the barrier, the whole team with it —
/// measured on the M3 Ultra, LFM3-1.7B, 20 workers, 2 322-token prompt, two
/// interleaved rounds of 6 requests: spin 930–1 086 tokens/s (35–49 % spread),
/// spin 256 then yield 1 473 (0.2–0.3 %), the decode unchanged (126–128).
/// Elsewhere, the spin barriers.
pub fn default_prefill_executor() -> crate::jit::execution::Executor {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        crate::jit::execution::Executor::SpmdHybrid
    } else {
        crate::jit::execution::Executor::SpmdSpin
    }
}

/// Pauses before a hybrid participant stops spinning: 256 on a Mac (see
/// [`default_prefill_executor`]), 512 elsewhere.
pub const fn default_hybrid_spin() -> u32 {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        256
    } else {
        512
    }
}

/// Does the default pool mix performance and efficiency cores? On a Mac
/// with fewer than eight performance cores the pool takes half of the
/// efficiency cores too ([`get_performance_cpu_count`]): the guided chunks
/// (`Knobs::task_guided`) are then the default. Elsewhere, `false`.
pub fn default_pool_includes_efficiency_cores() -> bool {
    #[cfg(target_os = "macos")]
    {
        use crate::core::isa::cores_of_level;
        if let (Some(p), Some(e)) = (cores_of_level(0), cores_of_level(1)) {
            return p < 8 && e >= 2;
        }
    }
    false
}

/// The rule of [`default_decode_workers`] for `p` performance cores.
pub const fn decode_workers_for(p: usize) -> Option<usize> {
    if p > 16 {
        Some(16)
    } else {
        None
    }
}

/// Physical P cores of an Intel hybrid (`/sys/devices/cpu_core/cpus`, one per
/// set of `thread_siblings`); 0 if the machine is not hybrid.
#[cfg(target_os = "linux")]
fn count_physical_p_cores_linux() -> usize {
    use std::collections::BTreeSet;
    let Ok(list) = std::fs::read_to_string("/sys/devices/cpu_core/cpus") else {
        return 0;
    };
    if !std::path::Path::new("/sys/devices/cpu_atom/cpus").exists() {
        return 0;
    }
    let mut seen = BTreeSet::new();
    for part in list.trim().split(',') {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<usize>().ok(), b.parse::<usize>().ok()),
            None => (part.parse::<usize>().ok(), part.parse::<usize>().ok()),
        };
        let (Some(a), Some(b)) = (a, b) else { continue };
        for cpu in a..=b {
            let siblings = std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
            .unwrap_or_default();
            let canonical = siblings
                .trim()
                .split(',')
                .filter_map(|s| s.split('-').next().and_then(|x| x.parse::<usize>().ok()))
                .min()
                .unwrap_or(cpu);
            seen.insert(canonical);
        }
    }
    seen.len()
}

/// The CPUs this process may run on, its affinity mask (a container's cpuset, `taskset`), or None
/// when it cannot be read.
#[cfg(target_os = "linux")]
pub fn allowed_cpus() -> Option<Vec<usize>> {
    // SAFETY: `cpu_set_t` is plain bits, all-zero valid; `sched_getaffinity` writes at most its
    // size.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    (r == 0).then(|| {
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| unsafe { libc::CPU_ISSET(c, &set) })
            .collect()
    })
}

/// Count physical cores on Linux by parsing thread_siblings_list.
#[cfg(target_os = "linux")]
fn count_physical_cores_linux() -> usize {
    use std::collections::BTreeSet;

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return 0;
    };

    let mut seen_cores = BTreeSet::new();

    let cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();

    for cpu_id in cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };

        // Parse siblings list and use the smallest as canonical
        let mut min_sibling = usize::MAX;
        for part in siblings_str.trim().split(',') {
            let part = part.trim();
            if let Some((start, _end)) = part.split_once('-') {
                if let Ok(s) = start.parse::<usize>() {
                    min_sibling = min_sibling.min(s);
                }
            } else if let Ok(n) = part.parse::<usize>() {
                min_sibling = min_sibling.min(n);
            }
        }

        if min_sibling != usize::MAX {
            seen_cores.insert(min_sibling);
        }
    }

    seen_cores.len()
}
