//! The AArch64 features, read from the kernel.
//!
//! macOS does not let user space read the `ID_AA64*` registers (an `mrs` of
//! them traps); it publishes the same answers as `hw.optional.arm.*` sysctls,
//! which is what the macOS probe reads, once. Linux publishes what it enabled
//! for user space in the auxiliary vector (`AT_HWCAP`, `AT_HWCAP2`, the bits of
//! `arch/arm64/include/uapi/asm/hwcap.h`), which is what the Linux probe reads.
//! Every ISA decision of the ARM target goes through these fields.

/// The CPU features the AArch64 target decides on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// Advanced SIMD (NEON).
    pub neon: bool,
    /// Half-precision arithmetic (FEAT_FP16).
    pub fp16: bool,
    /// Integer dot products (FEAT_DotProd).
    pub dotprod: bool,
    /// `fmlal` half to single (FEAT_FHM).
    pub fhm: bool,
    /// The BF16 extension: `bfdot`, `bfmmla`, `bfmlal[bt]`, `bfcvt[n]` (FEAT_BF16).
    pub bf16: bool,
    /// The IEEE-conformant BF16 dot products under `FPCR.EBF` (FEAT_EBF16).
    pub ebf16: bool,
    /// 8-bit integer matrix multiplies (FEAT_I8MM).
    pub i8mm: bool,
    /// Large System Extensions: `ldadd`, `cas`, `swp` (FEAT_LSE).
    pub lse: bool,
    /// Scalable Matrix Extension (FEAT_SME).
    pub sme: bool,
    /// SME2 (FEAT_SME2).
    pub sme2: bool,
    /// `bfmopa` into single-precision tiles (SME_B16F32).
    pub sme_b16f32: bool,
    /// `fmopa` single into single (SME_F32F32).
    pub sme_f32f32: bool,
    /// The largest streaming vector length, in bytes (0 without SME).
    pub sme_max_svl_bytes: u32,
    /// Bytes per cache line.
    pub cacheline: u32,
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let key = std::ffi::CString::new(name).ok()?;
    let mut value = [0u8; 8];
    let mut len = value.len();
    // SAFETY: `value` holds `len` writable bytes, the key is NUL-terminated.
    let rc = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    Some(match len {
        4 => u64::from(u32::from_ne_bytes(value[..4].try_into().ok()?)),
        8 => u64::from_ne_bytes(value),
        _ => return None,
    })
}

#[cfg(target_os = "macos")]
fn flag(name: &str) -> bool {
    sysctl_u64(name).is_some_and(|v| v != 0)
}

/// Read the features of the machine running this code.
#[cfg(target_os = "macos")]
pub fn detect() -> std::result::Result<Features, String> {
    if sysctl_u64("hw.optional.arm64").is_none() {
        return Err("the kernel publishes no hw.optional.arm64: not an AArch64 macOS".into());
    }
    Ok(Features {
        neon: flag("hw.optional.arm.AdvSIMD") || flag("hw.optional.neon"),
        fp16: flag("hw.optional.arm.FEAT_FP16"),
        dotprod: flag("hw.optional.arm.FEAT_DotProd"),
        fhm: flag("hw.optional.arm.FEAT_FHM"),
        bf16: flag("hw.optional.arm.FEAT_BF16"),
        ebf16: flag("hw.optional.arm.FEAT_EBF16"),
        i8mm: flag("hw.optional.arm.FEAT_I8MM"),
        lse: flag("hw.optional.arm.FEAT_LSE"),
        sme: flag("hw.optional.arm.FEAT_SME"),
        sme2: flag("hw.optional.arm.FEAT_SME2"),
        sme_b16f32: flag("hw.optional.arm.SME_B16F32"),
        sme_f32f32: flag("hw.optional.arm.SME_F32F32"),
        sme_max_svl_bytes: sysctl_u64("hw.optional.arm.sme_max_svl_b").unwrap_or(0) as u32,
        cacheline: sysctl_u64("hw.cachelinesize").unwrap_or(64) as u32,
    })
}

/// The number of cores of the performance level `level` (0 = the fastest),
/// or `None` when the kernel publishes no levels.
#[cfg(target_os = "macos")]
pub fn cores_of_level(level: u32) -> Option<usize> {
    sysctl_u64(&format!("hw.perflevel{level}.physicalcpu")).map(|n| n as usize)
}

// `arch/arm64/include/uapi/asm/hwcap.h`
#[cfg(target_os = "linux")]
mod hwcap {
    pub const ASIMD: u64 = 1 << 1;
    pub const ATOMICS: u64 = 1 << 8;
    pub const FPHP: u64 = 1 << 9;
    pub const ASIMDHP: u64 = 1 << 10;
    pub const ASIMDDP: u64 = 1 << 20;
    pub const ASIMDFHM: u64 = 1 << 23;
    pub const I8MM: u64 = 1 << 13;
    pub const BF16: u64 = 1 << 14;
    pub const SME: u64 = 1 << 23;
    pub const SME_B16F32: u64 = 1 << 28;
    pub const SME_F32F32: u64 = 1 << 29;
    pub const EBF16: u64 = 1 << 32;
    pub const SME2: u64 = 1 << 37;
}

/// Read the features of the machine running this code, from the auxiliary vector.
#[cfg(target_os = "linux")]
pub fn detect() -> std::result::Result<Features, String> {
    // SAFETY: `getauxval` only reads the process's auxiliary vector.
    let (h1, h2) = unsafe {
        (
            libc::getauxval(libc::AT_HWCAP) as u64,
            libc::getauxval(libc::AT_HWCAP2) as u64,
        )
    };
    if h1 & hwcap::ASIMD == 0 {
        return Err(format!("AT_HWCAP = {h1:#x} has no ASIMD bit: not an AArch64 Linux kernel that publishes its features"));
    }
    let has1 = |b: u64| h1 & b != 0;
    let has2 = |b: u64| h2 & b != 0;
    // SAFETY: `sysconf` takes no pointer; it answers -1 or 0 when it does not know.
    let line = unsafe { libc::sysconf(libc::_SC_LEVEL1_DCACHE_LINESIZE) };
    Ok(Features {
        neon: true,
        fp16: has1(hwcap::FPHP) && has1(hwcap::ASIMDHP),
        dotprod: has1(hwcap::ASIMDDP),
        fhm: has1(hwcap::ASIMDFHM),
        bf16: has2(hwcap::BF16),
        ebf16: has2(hwcap::EBF16),
        i8mm: has2(hwcap::I8MM),
        lse: has1(hwcap::ATOMICS),
        sme: has2(hwcap::SME),
        sme2: has2(hwcap::SME2),
        sme_b16f32: has2(hwcap::SME_B16F32),
        sme_f32f32: has2(hwcap::SME_F32F32),
        // The streaming vector length is read with `prctl(PR_SME_GET_VL)` when SME is used; not
        // read here.
        sme_max_svl_bytes: 0,
        cacheline: if line > 0 { line as u32 } else { 64 },
    })
}

/// Linux publishes no performance levels.
#[cfg(target_os = "linux")]
pub fn cores_of_level(_level: u32) -> Option<usize> {
    None
}
