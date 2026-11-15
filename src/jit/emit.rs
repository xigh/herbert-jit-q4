//! What the bf16 GEMM micro-kernel is asked for, whatever the target.
//!
//! A kernel call computes an `mr × nr` tile of `C`:
//!
//! ```text
//! C[i, j] = Σ_k A[i, k] · W[j, k]     i < mr, j < nr
//! ```
//!
//! with `A` bf16 (or f32) `[M, K]` row-major, `W` in the R16 layout
//! (`[N/16 blocks, K/2 pairs, 16 rows, 2 bf16]`, `model/weight.rs`) and `C`
//! f32 or bf16 `[M, N]` row-major. The ABI is three pointers, no context:
//! `(a, b, c)`. `K` and `N` are immediates of the kernel; every address is a
//! displacement from the three bases.
//!
//! The emitters live under `target/`: `arch/x86/emit.rs` (`vdpbf16ps`,
//! AVX-512), `arch/x86/emit_gemm_avx2.rs`, `arch/aarch64/gemm.rs`
//! (`bfdot`, NEON). This file holds the specification they share and the
//! target-independent part of its check; the register budget and the
//! addressing range belong to the target.

use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};

/// Options for kernel generation, measured: unrolling ×4 yields +0.2 % for
/// 2.4× the code on the 6×64 tile, hence the default of 1.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelOptions {
    /// Pairs of `K` processed per loop iteration.
    pub unroll: usize,
    /// Emit a `prefetcht0` on the `B` rows of the next iteration.
    pub prefetch: bool,
    /// Compute on the target's matrix engine (AArch64 SME: `bfmopa` into ZA)
    /// instead of its vector registers, with `A` pre-packed by pairs of `K`
    /// for sixteen rows (see [`packed_a_bytes`]). A target without one refuses it.
    pub matrix: bool,
    /// With `matrix`: compute on Apple's AMX (`matfp` bf16 → f32 into Z)
    /// instead of SME2, with `A` pre-packed by single `K` for thirty-two rows
    /// (see [`packed_a_amx_bytes`]). AArch64 macOS only.
    pub amx: bool,
    /// The variants of the AArch64 vector kernel; ignored by the x86 emitters.
    pub neon: NeonOptions,
    /// The variants of the RISC-V vector kernel; ignored elsewhere.
    pub rvv: RvvOptions,
}

impl Default for KernelOptions {
    fn default() -> Self {
        Self {
            unroll: 1,
            prefetch: true,
            matrix: false,
            amx: false,
            neon: NeonOptions::default(),
            rvv: RvvOptions::default(),
        }
    }
}

/// The multiply-accumulate of the AArch64 vector GEMM.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum NeonMac {
    /// `bfdot` by element: one instruction per pair and four columns, the
    /// pair summed then accumulated, each rounded to odd (measured); one
    /// per cycle on the M5.
    #[default]
    BfDot,
    /// `bfmlalb` then `bfmlalt` by element: two fused multiply-adds per pair
    /// and four columns, IEEE under `FPCR`; four per cycle on the M5. Not the
    /// bits of `bfdot`.
    BfMlal,
    /// The tail of the matrix path only (`NeonOptions::tail_mac`): SME2's
    /// `bfdot` into ZA, one instruction per row and pair for the four blocks
    /// of a tile (`gemm_sme::emit_rows`), the arithmetic of `bfmopa` — the
    /// bits of the tile. Refused by the vector kernels.
    SmeDot,
}

/// How the AArch64 vector GEMM loads a pair of `K` of a block of `W`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum NeonLoad {
    /// Two `ldp q` (32 bytes each).
    #[default]
    Ldp,
    /// One `ld1 {v.16b}` of four registers (64 bytes).
    Ld1x4,
}

/// The activations of the f16 AMX kernel (`--f16 --amx`, `arch/aarch64/gemm_amx.rs`), as the
/// packing node lays them out for it. The weights are f16 in every case.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum AmxF16A {
    /// The bf16 widened to f32 (exact), `fma32` with the weights read as f16 and widened by the
    /// unit: the bits of the `widen` kernel of `gemm_f16.rs` for every input but a NaN (the
    /// unit returns the default NaN).
    #[default]
    F32,
    /// Per block of thirty-two rows: f16 when every activation of the block comes back unchanged
    /// from f16, `matfp` f16; f32 and `fma32` otherwise. The bits of `F32`.
    Mixed,
    /// Rounded to f16, `matfp` f16: the bits of `widen` only while every activation is exact in
    /// f16 (from 2^-17 to 65504 in magnitude); outside, those of `widen` fed with the
    /// activations rounded to f16.
    F16,
}

/// The variants of the AArch64 vector GEMM, each a measurement arm; the
/// default is the kernel as first written. Part of the kernel cache key.
///
/// Despite the name, the weight-format fields (`w_int8`, `w_q4`, `q4_grid8`,
/// ...) are shared GEMM options: the x86-64 kernels read them too, through
/// `KernelOptions::neon`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NeonOptions {
    /// Columns of the decode tile (one row): 16, 32, 48 or 64, that many
    /// R16 blocks streamed per task.
    pub nr: usize,
    /// Bytes ahead of the current position of each block of `W` prefetched
    /// every 128 bytes, a multiple of 8 below 32768; 0 = no prefetch.
    pub prefetch: u16,
    /// Prefetch into L1 (`pldl1keep`) instead of L2 (`pldl2keep`).
    pub prefetch_l1: bool,
    /// The multiply-accumulate.
    pub mac: NeonMac,
    /// Independent accumulator sets over the pairs of a group (1 or 2): with
    /// two, the even pairs and the odd pairs of every group of four
    /// accumulate apart and are added at the end. Not the bits of one set.
    pub acc_sets: u8,
    /// The load of a pair of `W`.
    pub load: NeonLoad,
    /// Matrix path: residue rows `r = rows mod 16` with `1 ≤ r ≤ tail_rows`
    /// run on a vector kernel of `r` rows inside the same task as the tile,
    /// instead of a padded 16-row tile; 0 = the padded tile always.
    pub tail_rows: u8,
    /// The multiply-accumulate of the tail kernels: `bfdot` or SME2's `bfdot`
    /// into ZA (`SmeDot`), the bits of the tile, or `bfmlal`, faster than the
    /// vector `bfdot` on the residue rows but not the tile's bits.
    pub tail_mac: NeonMac,
    /// Residues up to this many rows take the `bfdot` tail whatever
    /// `tail_mac` (a second set of tail kernels): measured on the M5, one
    /// residue row is faster by `bfdot`, two to six by `bfmlal`. 0 = one set.
    pub tail_bfdot_max: u8,
    /// The kernel reads `A` in the packed layout of the matrix engine
    /// (`[K/2 pairs][16 rows][2]`) instead of row-major: the tail kernels.
    pub a_packed: bool,
    /// The weights of the projections are f16, k-major inside each pair of an R16 block
    /// (`[k0: 16 rows][k1: 16 rows]`, the same 64 bytes), not bf16 (`model/weight.rs`). AArch64
    /// without BF16; the kernels of `gemm_f16.rs`. The attention reads its own bf16 R16
    /// copies, untouched.
    pub w_f16: bool,
    /// With `w_f16`, the height of a tile from which the products are f16 `fmla .8h` into f16
    /// accumulators, added to the f32 ones every this many steps of k (a multiple of 8); 0 = every
    /// height widens f16 to f32 and accumulates in f32 (the bits of the bf16 `bfmlal` kernel).
    pub f16_flush: u16,
    /// With `w_f16` and the AMX: the form of the activations (`--amx-a`).
    pub amx_a: AmxF16A,
    /// The weights of the projections are int8 in groups of 32 values along k, one f32 scale per
    /// row and group (`jit/int8.rs`), the activations quantized to int8 by a node before the
    /// projections: `sdot`, with f32 scales applied per group. AArch64 with FEAT_DotProd;
    /// `gemm_int8.rs`. A shape that [`crate::jit::int8::gemm`] refuses keeps the bf16 weights.
    pub w_int8: bool,
    /// With `w_int8`, the weights are 4-bit (`jit/int8.rs`): two signed nibbles per byte, group of
    /// 32, f32 scale per row and group; `gemm_q4.rs`. The activations stay int8.
    pub w_q4: bool,
    /// With `w_q4`, the scale of a group is `signed_max / -8` (the `Q4_0` rule of ggml: the
    /// extreme value lands on the code −8), the default (grid 8); false, `absmax / 7` (grid 7,
    /// `--q4-grid7`).
    pub q4_grid8: bool,
    /// With `w_q4`, the language head (more than 100 000 rows) stays int8.
    pub q4_head_int8: bool,
    /// x86-64: emit the int8 / Q4 kernel in **SSE4.2 only** (`gemm_int8_sse42.rs`, `vpdpbusd`
    /// emulated with `pmaddubsw`/`pmaddwd`), for a CPU without AVX. Forced by `--q4-sse42` /
    /// `--int8-sse42`; ignored off x86.
    pub w_sse42: bool,
}

impl Default for NeonOptions {
    fn default() -> Self {
        Self {
            nr: 16,
            prefetch: 0,
            prefetch_l1: false,
            mac: NeonMac::BfDot,
            acc_sets: 1,
            load: NeonLoad::Ldp,
            tail_rows: 0,
            tail_mac: NeonMac::BfDot,
            tail_bfdot_max: 0,
            a_packed: false,
            w_int8: false,
            w_q4: false,
            w_sse42: false,
            q4_grid8: true,
            q4_head_int8: false,
            w_f16: false,
            f16_flush: 16,
            amx_a: AmxF16A::F32,
        }
    }
}

/// How many f32 lanes a RISC-V kernel accumulates over, where a sum folds
/// across the register: every lane VLMAX offers, or a fixed count whatever
/// VLEN. The bits of a folded sum depend on its lane count, so `Full` gives
/// bits that change with VLEN and `Lanes(n)` the same bits on every machine
/// whose VLMAX reaches `n`; both are measurement arms (the counts drop with
/// VLEN only under `Full`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum RvvWidth {
    /// All of VLMAX.
    #[default]
    Full,
    /// `n` lanes, a power of two ≥ 4 (fewer when VLMAX is smaller).
    Lanes(u16),
}

impl RvvWidth {
    /// The lane count on a machine whose widest f32 group holds `vlmax` lanes.
    pub fn lanes(self, vlmax: usize) -> usize {
        match self {
            RvvWidth::Full => vlmax,
            RvvWidth::Lanes(n) => vlmax.min(usize::from(n)),
        }
    }
}

/// The form of the RISC-V GEMM micro-kernel (`arch/riscv64/gemm.rs`).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum RvvGemm {
    /// `vlseg2e16` at `vl = 16` and `vfwmaccbf16.vf` per element of `A`: the
    /// same instructions at every VLEN ≥ 256.
    #[default]
    Seg,
    /// The whole vector length: consecutive pairs of `K` side by side in the
    /// lanes, `A` spread by `vrgather`, `vfwmaccbf16.vv`, the lanes folded
    /// at the end of the tile (`--rvv-width` sets how many).
    Wide,
}

/// The variants of the RISC-V vector kernels, each a measurement arm; the
/// default is the kernel as first written. Part of the kernel cache key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RvvOptions {
    /// The accumulation width of the folded sums.
    pub width: RvvWidth,
    /// Columns of the decode GEMM tile (one row): a multiple of 16, at most
    /// 112 (the vector registers of the `seg` form).
    pub nr: u16,
    /// The form of the GEMM micro-kernel.
    pub gemm: RvvGemm,
}

impl Default for RvvOptions {
    fn default() -> Self {
        Self {
            width: RvvWidth::Full,
            nr: 64,
            gemm: RvvGemm::Seg,
        }
    }
}

/// How a target runs the projections of one regime: the tile and the options
/// of its GEMM ([`crate::jit::arch::gemm_plan`]).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GemmPlan {
    /// The micro-kernel tile.
    pub tile: TileShape,
    /// Its options; `matrix` asks the lowering for a packing node before the
    /// projections of each activation buffer.
    pub opts: KernelOptions,
}

/// Rows of `A` per packed block of the matrix engine.
pub const PACK_ROWS: usize = 16;

/// Bytes of `A` packed for the matrix engine, `rows` rows of `K`: blocks of
/// [`PACK_ROWS`] rows, each `[K/2 pairs][16 rows][2 bf16]` — one pair of `K`
/// for sixteen rows in 64 contiguous bytes, as the R16 weights hold sixteen
/// columns. With an even `K` a block has the size of sixteen rows of `A`.
pub const fn packed_a_bytes(rows: usize, k: usize) -> usize {
    rows.div_ceil(PACK_ROWS) * PACK_ROWS * k.div_ceil(2) * 4
}

/// Rows of `A` per packed block of Apple's AMX.
pub const AMX_ROWS: usize = 32;

/// Bytes of `A` packed for Apple's AMX, `rows` rows of an even `K`: blocks of
/// [`AMX_ROWS`] rows, each `[K][32 rows]` — one `K` for thirty-two rows in 64
/// contiguous bytes (bf16 or f16), the Y operand of one `matfp`, or in 128
/// (f32, [`AmxF16A::F32`]), two Y operands of `fma32`. A block has
/// [`AMX_ROWS`] times [`amx_row_bytes`].
pub const fn packed_a_amx_bytes(rows: usize, k: usize, neon: NeonOptions) -> usize {
    rows.div_ceil(AMX_ROWS) * AMX_ROWS * amx_row_bytes(k, neon)
}

/// The share of one row of `A` in a packed AMX block: two bytes per `K`
/// (bf16, or f16 under [`AmxF16A::F16`]), four (f32), or under
/// [`AmxF16A::Mixed`] four plus two — the block opens on 64 bytes whose first
/// word says the form of its data (0 f16, 1 f32), the data room that of f32.
pub const fn amx_row_bytes(k: usize, neon: NeonOptions) -> usize {
    let k = k.next_multiple_of(2);
    if !neon.w_f16 {
        return k * 2;
    }
    match neon.amx_a {
        AmxF16A::F16 => k * 2,
        AmxF16A::F32 => k * 4,
        AmxF16A::Mixed => k * 4 + 2,
    }
}

/// Bytes of the packed `A` a plan's options ask for, `rows` rows of `K`.
pub const fn packed_bytes(opts: KernelOptions, rows: usize, k: usize) -> usize {
    if opts.neon.w_int8 {
        crate::jit::int8::a_bytes(rows, k)
    } else if opts.amx {
        packed_a_amx_bytes(rows, k, opts.neon)
    } else {
        packed_a_bytes(rows, k)
    }
}

/// Memory format of `A`, and — when it is in f32 — the rounding used to
/// bring it back to bf16 **in register**.
///
/// `vdpbf16ps` requires its two operands in bf16: an `A` f32 is **forced**
/// to be rounded, the only question is how: truncation or round to nearest
/// even.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum AFormat {
    /// `A` is already in bf16 `[M, K]`: direct diffusion of the pair.
    #[default]
    Bf16,
    /// `A` in f32, **truncated** (`shr 16`) — at most 1 ulp bf16.
    ///
    /// The bits of an `A` truncated to bf16 before the kernel: same
    /// operands, same accumulation order, same output bits.
    F32Trunc,
    /// `A` in f32, **rounded to nearest even** (`vcvtneps2bf16`) — 0.5 ulp.
    ///
    /// More accurate than truncation, so **not** the same bits: the digits
    /// move. The rounding of an `A` bf16 of `m × k` materialized beforehand,
    /// without that buffer.
    F32Rne,
}

impl AFormat {
    /// Bytes consumed in `A` by a pair of `K`.
    pub const fn pair_bytes(self) -> usize {
        match self {
            AFormat::Bf16 => 4,
            AFormat::F32Trunc | AFormat::F32Rne => 8,
        }
    }

    /// Bytes per element of `A`.
    pub const fn elem_bytes(self) -> usize {
        self.pair_bytes() / 2
    }

    /// Is `A` read in f32?
    pub const fn is_f32(self) -> bool {
        !matches!(self, AFormat::Bf16)
    }
}

/// Everything the emitter needs to produce a kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MicroKernelSpec {
    /// Lines of `C` calculated by this call (1..=mr of the plan).
    pub mr: usize,
    /// Columns of `C` calculated by this call, multiple of 16.
    pub nr: usize,
    /// Pairs of `K` (= `ceil(K/2)`), immediate — **padding included**.
    pub k_pairs: usize,
    /// `K` is odd: the last of the `k_pairs` pairs has only one valid element in `A`. See the
    /// module preamble. Distance in bytes between two lines of `A` (= `2·K` in bf16, `4·K` in
    /// f32).
    pub k_odd: bool,
    /// Byte distance between two rows of `A` (= `2·K` in bf16, `4·K` in f32).
    pub lda_bytes: usize,
    /// Memory layout of `A`, and how it is rounded when f32: the f32 pair is
    /// brought to bf16 in a register while loading, without an `m × k` buffer
    /// (the DeltaNet projections feed f32 activations to the same weights).
    pub a_fmt: AFormat,
    /// Distance in bytes between two rows of `C` (`4·N` in f32, `2·N` in bf16).
    pub ldc_bytes: usize,
    /// **Valid** columns in the last group of 16 of this tile, in
    /// `1..=16`. `16` = full tile, no mask emitted. See the preamble.
    pub n_tail: usize,
    /// Is `C` written in bf16?
    ///
    /// The engine invariant: f32 only in the accumulators, bf16 in memory.
    ///
    /// Accumulation does not change — it remains in f32 in the zmm registers, which is
    /// exactly what the rule allows. Only the **epilogue** changes: instead
    /// of writing 16 f32 per register, it converts to bf16
    /// (`vcvtneps2bf16`, AVX512-BF16) and writes only 32 bytes. Direct
    /// consequence: the next consumer has nothing left to convert, whereas it
    /// was calling `cvt_f32_to_bf16` on its entire input until now.
    pub out_bf16: bool,
    /// Generation options.
    pub opts: KernelOptions,
    /// The length of `K` is given at each call, not at emission: the loop
    /// runs `x15` groups of four pairs (set by the caller, at least one),
    /// and `k_pairs` is only the distance between two R16 blocks of `W`, in
    /// pairs — a `W` reserved for more than it holds, like the `Vᵀ` of the
    /// KV cache that the block attention reads over the keys seen so far
    /// (`arch/aarch64/sdpa_block.rs`). `K` even, `k_pairs` a multiple of
    /// four. AArch64 only; the x86 emitters refuse it.
    pub k_dynamic: bool,
}

/// Signature of the generated code. The third pointer targets `f32` or `u16`
/// according to [`MicroKernelSpec::out_bf16`] — hence the opaque pointer.
///
/// The calling convention of generated code (`sysv64` on x86-64 on every
/// operating system, AAPCS64 on AArch64): see `arch::generated_fn!`.
pub type KernelFn = crate::jit::arch::generated_fn!(*const u16, *const u16, *mut std::ffi::c_void);

/// Distance in bytes between two consecutive R16 blocks of `B`.
pub const fn b_block_bytes(k_pairs: usize) -> usize {
    // [k_pairs, 16 rows_of, 2 values] × 2 bytes
    k_pairs * ROWS_INTERLEAVED * 2 * 2
}

impl MicroKernelSpec {
    /// Checks the shape against the target: the tile fits its registers, and
    /// every offset the generated code carries fits its addressing forms.
    pub fn check(&self) -> Result<(), Decline> {
        if self.k_pairs == 0 {
            return Err(Decline::Empty);
        }
        if self.n_tail == 0 || self.n_tail > ROWS_INTERLEAVED {
            return Err(Decline::PartialRowBlock);
        }
        if self.k_dynamic && (self.k_odd || !self.k_pairs.is_multiple_of(4)) {
            return Err(Decline::OddK);
        }
        crate::jit::arch::check_gemm(self)
    }
}

/// Generates the micro-kernel described by `spec` for the target of this
/// build, after [`MicroKernelSpec::check`].
pub fn emit(spec: &MicroKernelSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::arch::emit_gemm(spec)
}
