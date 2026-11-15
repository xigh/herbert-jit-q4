//! The GatedDeltaNet recurrence, whatever the target: one V head, one
//! position.
//!
//! `s` : `[hkd · hvd]` f32 (bf16 with `RecurrenceSpec::state16`), persistent,
//! `s[kd·hvd + vd]`. The order is the specification:
//!
//! ```text
//! 1. s ← decay·s ;  kv[vd] = Σ_kd s[kd·hvd+vd] · k[kd]
//!    δ[vd] = (v[vd] − kv[vd]) · beta
//! 2. s[kd·hvd+vd] += k[kd]·δ[vd]
//!    out[vd] = Σ_kd s[kd·hvd+vd] · q[kd] · scale
//! ```
//!
//! ABI: `fn(s: *mut f32, out: *mut f32, q: *const f32, k: *const f32, v: *const f32,
//! hkd: usize, beta: f32, decay: f32, scale: f32)`. `hkd` is a parameter,
//! `hvd` an immediate. The emitters: `arch/x86/emit_recurrence.rs`
//! (AVX-512), `arch/aarch64/deltanet.rs`, `arch/riscv64/deltanet.rs`.

use crate::jit::ir::Decline;

/// Shape of a recurrence kernel: the head width V, **immediate**, and
/// the choice of FMA — another blob, hence another cache key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecurrenceSpec {
    /// `head_v_dim` — 128 on Qwen3.5. Multiple of 16, at most 192.
    pub hvd: usize,
    /// `vfmadd231ps` for the three dot products (header). `false` = the bits
    /// of the scalar.
    pub fma: bool,
    /// AArch64: the first pass does not store the decayed state; the second
    /// pass recomputes `s · decay` from the state as read — the same product,
    /// the same bits, one store of the state per position instead of two.
    /// Refused by the x86 emitter.
    pub store_once: bool,
    /// AArch64: the state loaded and stored by pairs of registers (`ldp`/`stp`
    /// of two `q`): half the memory instructions, the same bits. Refused by x86.
    pub pairs: bool,
    /// AArch64: `q · scale` computed once per position into a scratch on the
    /// stack (`hkd ≤ 512`) instead of once per slice of the value head: the
    /// same product, the same bits. Refused by x86.
    pub q_hoist: bool,
    /// AArch64: lanes of the value head accumulated by one loop of the first
    /// pass, 32 or 64 (sixteen accumulators, the output ones borrowed); the
    /// second pass keeps slices of 32. The same bits. Refused by x86 unless 32.
    pub pass1_lanes: u8,
    /// Lanes of the value head this call computes, from the pointers it is
    /// given; 0 = all `hvd`. The row stride of the state stays `hvd`: a head
    /// cut into slices of independent columns, one task each. The same bits.
    /// Refused by x86 unless 0.
    pub width: usize,
    /// The state kept in bf16, the engine's 16-bit format, as the K/V cache
    /// (`Knobs::deltanet_state_f32` off): `s` is `[hkd · hvd]` bf16, widened
    /// exactly as it is loaded, rounded to nearest even as the second pass
    /// stores it, after `out` has read it; the first pass stores nothing
    /// (the second recomputes `s · decay`, the same product). `false`: f32.
    pub state16: bool,
    /// Mamba2 (Granite 4) instead of the delta rule, one pass: `s ← decay·s + (k·beta) ⊗ v`, `out
    /// = Σ_kd s·q`, then `out += v · scale`, the skip `D` passed as the scale (`beta` is the
    /// step `dt`, `k` is B, `q` is C, `v` is x). x86 only for now.
    pub mamba: bool,
}

impl RecurrenceSpec {
    /// The form as first written: two stores, single registers, 32 lanes, the whole head.
    pub const fn new(hvd: usize, fma: bool) -> RecurrenceSpec {
        RecurrenceSpec {
            hvd,
            fma,
            store_once: false,
            pairs: false,
            q_hoist: false,
            pass1_lanes: 32,
            width: 0,
            state16: false,
            mamba: false,
        }
    }
}

/// `fn(s, out, q, k, v, hkd, beta, decay, scale)`.
// The calling convention of generated code (`sysv64` on x86-64 on every
// operating system, AAPCS64 on AArch64): see `arch::generated_fn!`.
pub type RecurrenceFn = crate::jit::arch::generated_fn!(
    *mut f32,
    *mut f32,
    *const f32,
    *const f32,
    *const f32,
    usize,
    f32,
    f32,
    f32,
);

impl RecurrenceSpec {
    /// Emissible shapes: `hvd` a non-zero multiple of 16 (whole blocks, no
    /// remainder), then the register budget of the target.
    pub fn check(&self) -> Result<(), Decline> {
        if self.hvd == 0 {
            return Err(Decline::Empty);
        }
        if !self.hvd.is_multiple_of(16) {
            return Err(Decline::PartialRowBlock);
        }
        if !matches!(self.pass1_lanes, 32 | 64)
            || self.width > self.hvd
            || (self.width > 0 && !self.width.is_multiple_of(32))
        {
            return Err(Decline::PartialRowBlock);
        }
        crate::jit::arch::check_recurrence(self)
    }

    /// 16 f32 ZMM covering `hvd`.
    pub const fn blocks16(&self) -> usize {
        self.hvd / 16
    }
}

/// Generates the kernel of `spec` for the target of this build.
pub fn emit_recurrence(spec: &RecurrenceSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::arch::emit_recurrence(spec)
}
