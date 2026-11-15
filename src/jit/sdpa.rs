//! The attention driver: the shape policy, compilation and cache of the SDPA
//! kernels — one per `(head_dim, sink, form, exp)` for decode, one block
//! kernel of `rows` queries for prefill.
//!
//! What the knobs decide (`knobs.rs`): the form (chained, split, fused), the
//! exponential, the rows of a prefill block, the QK group and the pairing.
//! A shape the emitter declines is cached as declined and the caller falls
//! back to the decode kernel called row by row.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_sdpa::{
    emit_sdpa, emit_sdpa_prefill, Pass, SdpaHeadFn, SdpaKernelSpec, SdpaPrefillFn, SdpaPrefillSpec,
    SdpaQkFn, SdpaSoftmaxFn, SdpaSvFn, SDPA_PREFILL_MAX_ROWS,
};
use crate::jit::gemm::is_supported;
use crate::jit::ir::{Decline, ExpStrategy, SdpaForm, SdpaShape};
use crate::jit::KERNEL_ALIGN;
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// The SDPA form in force ([`crate::jit::knobs::Knobs::sdpa_form`]).
pub fn sdpa_form() -> SdpaForm {
    crate::jit::knobs::knobs().sdpa_form
}

/// The exponential strategy in force ([`crate::jit::knobs::Knobs::sdpa_exp`]).
pub fn sdpa_exp() -> ExpStrategy {
    crate::jit::knobs::knobs().sdpa_exp
}

/// Spill every SV accumulator ([`crate::jit::knobs::Knobs::sdpa_sv_spill_all`]).
pub fn sdpa_sv_spill_all() -> bool {
    crate::jit::knobs::knobs().sdpa_sv_spill_all
}

/// Queries per prefill block ([`crate::jit::knobs::Knobs::sdpa_rows`]), capped
/// at what the emitter supports; `0` = the decode kernel row by row.
pub fn sdpa_rows() -> usize {
    crate::jit::knobs::knobs()
        .sdpa_rows
        .min(SDPA_PREFILL_MAX_ROWS)
}

/// 16-dimension blocks of the SV pass ([`crate::jit::knobs::Knobs::sdpa_dim_blocks`]);
/// `0` = the emitter's default, [`SdpaPrefillSpec::widest_dim_blocks`].
pub fn sdpa_dim_blocks() -> usize {
    crate::jit::knobs::knobs().sdpa_dim_blocks.min(16)
}

/// Positions per tile of the block kernel's SV pass (`SdpaPrefillSpec::sv_tile`), from
/// `HERBERT_SDPA_SV_TILE`: unset or `0` the shape's default, `1` no tile, `T` tiles of `T`
/// positions. A measurement arm.
pub fn sdpa_sv_tile() -> usize {
    static TILE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| {
        std::env::var("HERBERT_SDPA_SV_TILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// The prefetch distance imposed on both kernels, or `None` for their own
/// defaults ([`crate::jit::knobs::Knobs::sdpa_prefetch`]).
pub fn sdpa_pf_dist() -> Option<usize> {
    crate::jit::knobs::knobs().sdpa_prefetch
}

/// A compiled **prefill** SDPA: a block of `rows` queries.
pub struct CompiledSdpaPrefill {
    spec: SdpaPrefillSpec,
    /// Keeps the executable page alive.
    _exec: Mmap,
    run: SdpaPrefillFn,
    code_bytes: usize,
}

// SAFETY: same reasoning as for `CompiledSdpa` — RX page immutable, no
// global state in the generated code.
unsafe impl Send for CompiledSdpaPrefill {}
unsafe impl Sync for CompiledSdpaPrefill {}

impl CompiledSdpaPrefill {
    /// The bytes of the block kernel, copied into a traversal (AArch64 and
    /// x86; RISC-V has no block kernel).
    #[cfg_attr(target_arch = "riscv64", allow(dead_code))]
    pub(crate) fn lowering_bytes(&self) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.run as *const u8, self.code_bytes).to_vec() }
    }

    /// Compiled specification.
    pub fn spec(&self) -> SdpaPrefillSpec {
        self.spec
    }

    /// Requests processed per call.
    pub fn rows(&self) -> usize {
        self.spec.rows
    }

    /// 16-dimension blocks accumulated together by the SV pass.
    pub fn dim_blocks(&self) -> usize {
        self.spec.sv_dim_blocks()
    }

    /// Machine code bytes produced.
    pub fn code_bytes(&self) -> usize {
        self.code_bytes
    }

    /// The scratch of one block, in f32: one slot per participant.
    pub fn scratch_len(&self) -> usize {
        self.spec.scratch_len()
    }

    /// Required length of the scores buffer, in f32, for `past` positions
    /// already cached.
    pub fn scores_len(&self, past: usize) -> usize {
        self.spec.scores_len(past)
    }
}

/// The prefill kernel process for this shape. `Ok(None)` = fallback to the
/// decode kernel called line by line.
pub fn prefill_kernel(
    shape: SdpaShape,
    rows: usize,
    row_stride: usize,
    out_row_stride: usize,
    kv_capacity: usize,
) -> Result<Option<Arc<CompiledSdpaPrefill>>> {
    if !is_supported() {
        // x86-64 with AVX2 alone: the AVX2 block kernel
        // (`arch/x86/emit_sdpa_avx2.rs`), the decode kernel's sequence row
        // by row with `K` and `V` read once for several rows — the bits of
        // the decode kernel called row by row, which serves without it. One
        // exp, no QK group nor pairing: the AVX2 decode kernel has none.
        #[cfg(target_arch = "x86_64")]
        if crate::jit::emit_sdpa_avx2::is_supported() {
            return get_or_compile_prefill(SdpaPrefillSpec {
                shape,
                rows,
                row_stride,
                out_row_stride,
                exp: ExpStrategy::Inline,
                pf_dist: 0,
                dim_blocks: 0,
                qk_group: 0,
                qk_pair: 1,
                kv_capacity,
                full: false,
                window: 0,
                sv_tile: 0,
            });
        }
        return Ok(None);
    }
    // AArch64: the block kernel is the two-GEMM attention of
    // `arch/aarch64/sdpa_block.rs`, sixteen queries per block, under
    // `Knobs::attn_block`; without it the decode kernel serves a block row
    // by row, with the same bits (the QK of a position does not depend on
    // its neighbours, `arch/aarch64/sdpa.rs`). RISC-V takes the same
    // branch: `attn_block` is off there and its block kernel not written.
    if cfg!(any(target_arch = "aarch64", target_arch = "riscv64")) {
        if !crate::jit::knobs::knobs().attn_block {
            return Ok(None);
        }
        let spec = SdpaPrefillSpec {
            shape,
            rows: crate::jit::emit_sdpa::SDPA_BLOCK_ROWS,
            row_stride,
            out_row_stride,
            exp: ExpStrategy::Inline,
            pf_dist: 0,
            dim_blocks: 0,
            qk_group: 0,
            qk_pair: 1,
            kv_capacity,
            full: false,
            window: 0,
            sv_tile: 0,
        };
        return get_or_compile_prefill(spec);
    }
    let spec = SdpaPrefillSpec {
        shape,
        rows,
        row_stride,
        out_row_stride,
        exp: sdpa_exp(),
        pf_dist: sdpa_pf_dist().unwrap_or(crate::jit::emit_sdpa::SDPA_PF_DIST_PREFILL),
        dim_blocks: sdpa_dim_blocks(),
        qk_group: sdpa_qk_group(),
        qk_pair: sdpa_qk_pair(),
        kv_capacity,
        full: false,
        window: 0,
        sv_tile: sdpa_sv_tile(),
    };
    // The default pairing is removed when the shape does not support it.
    // Without this folding, `check` would reject the entire spec and the caller
    // would fall back to the line-by-line decode kernel — 17 to 19% slower than
    // the non-paired block. What was **requested** is not removed (see
    // `sdpa_qk_pair_raw`): its rejection is propagated as is.
    if spec.qk_pair > 1 && spec.check().is_err() && sdpa_qk_pair_raw().is_none() {
        let without = SdpaPrefillSpec { qk_pair: 1, ..spec };
        if without.check().is_ok() {
            if crate::jit::options::debug_plan() {
                eprintln!(
                    "DEBUG [backend_jit] sdpa prefill head_dim={} rows={} qk={} — \
                     pairing p={} removed (registers), fallback p=1",
                    shape.head_dim, rows, spec.qk_group, spec.qk_pair,
                );
            }
            return get_or_compile_prefill(without);
        }
    }
    get_or_compile_prefill(spec)
}

/// [`prefill_kernel`] without the causal mask (`SdpaPrefillSpec::full`): every row of a block sees
/// the same keys, `qk_group` the QK group. `window > 0`: the bidirectional window `|q − k| <
/// window` (EmbeddingGemma 2's sliding layers).
///
/// - x86-64 with AVX-512: blocks of `rows`, the sixth argument the count `n` of keys; with a
///   window, the tile's `past` and the end of the keys in `rax` (`SdpaPrefillSpec::window`).
/// - x86-64 with AVX2 alone: the AVX2 block (`arch/x86/emit_sdpa_avx2.rs`), same rows and calling
///   convention, one exp, no QK group (`qk_group` ignored).
/// - AArch64 (under `Knobs::attn_block`): the two-GEMM block of `arch/aarch64/sdpa_block.rs`,
///   sixteen rows, on the R16 copies of the cache (written by the KV append under `KvSpec::r16`,
///   `kv_capacity` a multiple of sixteen: [`full_kv_capacity`]); `rows` only says whether a block
///   is wanted (0: the row kernel).
///
/// `Ok(None)` elsewhere: the row kernel then serves.
pub fn prefill_kernel_full(
    shape: SdpaShape,
    rows: usize,
    row_stride: usize,
    out_row_stride: usize,
    kv_capacity: usize,
    qk_group: u8,
    window: usize,
) -> Result<Option<Arc<CompiledSdpaPrefill>>> {
    if rows == 0 {
        return Ok(None);
    }
    if cfg!(target_arch = "aarch64") {
        if !is_supported() || !crate::jit::knobs::knobs().attn_block {
            return Ok(None);
        }
        return get_or_compile_prefill(SdpaPrefillSpec {
            shape,
            rows: crate::jit::emit_sdpa::SDPA_BLOCK_ROWS,
            row_stride,
            out_row_stride,
            exp: ExpStrategy::Inline,
            pf_dist: 0,
            dim_blocks: 0,
            qk_group: 0,
            qk_pair: 1,
            kv_capacity,
            full: true,
            window,
            sv_tile: 0,
        });
    }
    // x86-64 with AVX2 alone: the AVX2 block kernel without the causal mask, with or without the
    // bidirectional window (`arch/x86/emit_sdpa_avx2.rs`), the decode kernel's sequence row by
    // row and its bits; one exp, no QK group nor pairing (the caller's `qk_group` is the
    // AVX-512 row kernel's), the same calling convention as AVX-512.
    #[cfg(target_arch = "x86_64")]
    if !is_supported() && crate::jit::emit_sdpa_avx2::is_supported() {
        return get_or_compile_prefill(SdpaPrefillSpec {
            shape,
            rows,
            row_stride,
            out_row_stride,
            exp: ExpStrategy::Inline,
            pf_dist: 0,
            dim_blocks: 0,
            qk_group: 0,
            qk_pair: 1,
            kv_capacity,
            full: true,
            window,
            sv_tile: 0,
        });
    }
    if !is_supported() || !cfg!(target_arch = "x86_64") {
        return Ok(None);
    }
    let spec = SdpaPrefillSpec {
        shape,
        rows,
        row_stride,
        out_row_stride,
        exp: sdpa_exp(),
        pf_dist: sdpa_pf_dist().unwrap_or(crate::jit::emit_sdpa::SDPA_PF_DIST_PREFILL),
        dim_blocks: sdpa_dim_blocks(),
        qk_group,
        qk_pair: sdpa_qk_pair(),
        kv_capacity,
        full: true,
        window,
        sv_tile: sdpa_sv_tile(),
    };
    if spec.qk_pair > 1 && spec.check().is_err() && sdpa_qk_pair_raw().is_none() {
        return get_or_compile_prefill(SdpaPrefillSpec { qk_pair: 1, ..spec });
    }
    get_or_compile_prefill(spec)
}

/// The positions an unmasked attention over `rows` keys reserves per KV head, and whether its KV
/// append writes the R16 copies: on AArch64 under `Knobs::attn_block`, `rows` rounded up to sixteen
/// and the copies (the block kernel of [`prefill_kernel_full`] reads them); elsewhere `rows`, none.
pub fn full_kv_capacity(rows: usize) -> (usize, bool) {
    if cfg!(target_arch = "aarch64") && is_supported() && crate::jit::knobs::knobs().attn_block {
        (rows.next_multiple_of(16).max(16), true)
    } else {
        (rows, false)
    }
}

/// The scratch, in f32, of one participant of an unmasked attention over `rows` keys with heads of
/// at most `head_dim`: the row kernel's scores or a block's, whichever is larger
/// ([`prefill_kernel_full`]).
pub fn full_scratch_f32(rows: usize, head_dim: usize) -> usize {
    let rows_kernel =
        rows.next_multiple_of(16).max(16) * crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS;
    // The x86 block with the bidirectional window: up to fifteen columns of padding past its keys,
    // one more block of sixteen per row (`SdpaPrefillSpec::scratch_len`).
    let rows_kernel = rows_kernel + 16 * crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS;
    let (capacity, r16) = full_kv_capacity(rows);
    // The AArch64 block: scores and P over the capacity, three tiles of queries
    // (`sdpa_block::scratch_f32`).
    let block = if r16 {
        (16 * capacity * 4 + 16 * capacity * 2 + 3 * 16 * head_dim * 2) / 4
    } else {
        0
    };
    rows_kernel.max(block)
}

/// QK group size for this process: `0` (no), `8` or `16`.
///
/// **Default: 8.** What decided it was not speed
/// alone but the **resource**: at equal µops per score (8.25
/// vs 8.2), the group of 8 holds 14 live zmm vs 22, weighs 8 145
/// bytes of code at prefill vs 11 488, and admits shapes that 16 refuses
/// due to lack of registers. It is never slower — equality at short context,
/// −3.3 % at `past` 4 096 — and the whole kernel gains 17 to 19 % on the path
/// line by line (µbench, arms alternated in the same process).
///
/// **This default changes the summation order, and therefore the bits.** The
/// single chain replaces the four chains: the deviation is bounded to **one
/// bf16 step** (96.8 % of scores are bit-identical), with no measurable end-to-
/// end degradation (logit deviation of 1.24 e−2 versus 1.30 e−2, one argmax
/// disagreement instead of two). The **block ↔ repeated decode** equivalence,
/// however, remains intact.
///
/// **A single knob for both kernels**, prefill and decode, and this is not
/// a convenience: it is the only setting that changes the order of summation, hence
/// the only one that can break the bit-for-bit equality of the block with repeated decode.
/// It holds as long as both ends change together; setting them separately
/// would break it without either kernel being wrong.
pub fn sdpa_qk_group() -> u8 {
    qk_group_raw().unwrap_or(8)
}

/// Requests that **share each `K` load** in the prefill group loop: `1` (`K` as
/// a memory operand of each request's `vdpbf16ps`) or more.
///
/// **Default: 2.** What settled it was a resource: the grouped QK pass was
/// losing **61 % of its cycles in load queue tokens**
/// (`PMCx0AE` bit 1, on a Zen 5) because it rereads `K` from memory for
/// each request. At `p = 2` the segment is loaded once into a register and
/// read by two FMAs: the stall **disappears** (3.05 → 0.00 cycles per pair),
/// cycles drop from 4.81 to **3.08** for +22 % instructions, IPC rises
/// from 1.87 to 3.58. **QK pass −32 to −35 %, whole kernel −12.5 to −17.7 %**
/// (µbench, alternating arms in the same process).
///
/// **No change in summation order**: each accumulator receives the
/// same segments in the same order, only the operand provenance changes.
/// Bit-for-bit equality with `p = 1` and with repeated decode holds
/// — this defect therefore changes **no output bit**, unlike
/// [`sdpa_qk_group`].
///
/// It is, however, a **resource** decision: 27 zmm out of 32 at
/// `head_dim` 128 and `g = 8`, and the emitter declines beyond (`head_dim` 256,
/// `g = 16`, outside the group loop). The fallback is written in
/// [`prefill_kernel`]: see [`sdpa_qk_pair_raw`] for the rule.
pub fn sdpa_qk_pair() -> u8 {
    sdpa_qk_pair_raw().unwrap_or(2)
}

/// The pairing imposed by [`crate::jit::knobs::Knobs::sdpa_qk_pair`], or `None`.
///
/// A **requested** pairing that does not fit in registers must propagate its
/// refusal, because a measuring arm that folded silently would disguise the
/// control number as a result. A pairing **inherited from the default**
/// withdraws for the shape that cannot carry it — otherwise the whole block
/// kernel would vanish at `head_dim` 256 in favour of the row-by-row path,
/// which is 17 to 19 % slower.
pub fn sdpa_qk_pair_raw() -> Option<u8> {
    crate::jit::knobs::knobs().sdpa_qk_pair
}

/// The QK group imposed by [`crate::jit::knobs::Knobs::sdpa_qk_group`], or `None`:
/// what [`qk_group_for`] treats as requested rather than inherited.
pub fn qk_group_raw() -> Option<u8> {
    crate::jit::knobs::knobs().sdpa_qk_group
}

/// The **effective** group size for a given shape.
///
/// The fused form interlaces its QK with the softmax and cannot hold a group
/// of positions on the fly: it refuses it ([`Decline::Qk16InFused`]), and
/// this refusal is deliberate — a measuring arm that would fold silently
/// would render the control number disguised as a result.
///
/// **But once it becomes the default, the group must no longer break a shape that no one
/// asked to change.** Hence the rule, which fits in a single sentence: *what the
/// user asked for is opposed to them, what they did not ask for is
/// removed.*
///
/// - size set **explicitly** → the refusal propagates: whoever requests `fused` **and** a group
///   must learn that they are incompatible, not receive something other than what they asked for;
/// - size coming from the **default** → it is removed for this form alone.
pub fn qk_group_for(form: SdpaForm, group: u8, explicit: bool) -> u8 {
    if form == SdpaForm::Fused && !explicit {
        0
    } else {
        group
    }
}

/// Compile `spec`, or return the result already in cache. `Ok(None)` =
/// form rejected, the caller falls back to the decode core.
///
/// Entry point **without environment policy**: both exp strategies can be
/// compiled within the same process.
pub fn get_or_compile_prefill(spec: SdpaPrefillSpec) -> Result<Option<Arc<CompiledSdpaPrefill>>> {
    if spec.check().is_err() {
        return Ok(None);
    }
    let cached = prefill_cache().read().unwrap().get(&spec).cloned();
    if let Some(hit) = cached {
        return Ok(hit);
    }
    let built = Some(Arc::new(map_prefill(spec, emit_sdpa_prefill(&spec)?)?));
    if crate::jit::options::debug_plan() {
        if let Some(k) = built.as_deref() {
            eprintln!(
                "DEBUG [backend_jit] sdpa prefill head_dim={} rows={} nb={}  qk={} \
                 p={} pf={} stride={} exp={} bytes={}",
                spec.shape.head_dim,
                spec.rows,
                spec.sv_dim_blocks(),
                spec.qk_group,
                spec.qk_pair(),
                spec.pf_dist,
                spec.row_stride,
                spec.exp.name(),
                k.code_bytes(),
            );
        }
    }
    let mut guard = prefill_cache().write().unwrap();
    Ok(guard.entry(spec).or_insert(built).clone())
}

type PrefillCache = RwLock<HashMap<SdpaPrefillSpec, Option<Arc<CompiledSdpaPrefill>>>>;

fn prefill_cache() -> &'static PrefillCache {
    static CACHE: OnceLock<PrefillCache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn map_prefill(spec: SdpaPrefillSpec, blob: Vec<u8>) -> Result<CompiledSdpaPrefill> {
    let exec = crate::jit::arch::map_exec(blob.len().max(KERNEL_ALIGN), |map| {
        map[..blob.len()].copy_from_slice(&blob)
    })
    .map_err(|e| HerbertError::Backend(format!("executable mapping: {e}")))?;
    crate::jit::perf_map::register(exec.as_ptr(), blob.len(), || {
        format!("sdpa_prefill_{}", compact(&format!("{spec:?}")))
    });
    // SAFETY: the blob was just written into this page, made executable, and
    // `exec` retains ownership of it. The signature is the one emitted by
    // `emit_sdpa_prefill`.
    let run = unsafe { std::mem::transmute::<*const u8, SdpaPrefillFn>(exec.as_ptr()) };
    Ok(CompiledSdpaPrefill {
        spec,
        _exec: exec,
        run,
        code_bytes: blob.len(),
    })
}

/// A compiled decode SDPA for a given shape.
pub struct CompiledSdpa {
    spec: SdpaKernelSpec,
    /// Keeps the executable page alive.
    _exec: Mmap,
    /// Shapes `Chained` and `Fused`: a single blob per head.
    head: Option<SdpaHeadFn>,
    /// `Split` shape: three blobs chained on the Rust side.
    split: Option<(SdpaQkFn, SdpaSoftmaxFn, SdpaSvFn)>,
    kernel_lengths: Vec<usize>,
    code_bytes: usize,
}

// SAFETY : the function pointers target a shared and immutable RX page; `Mmap`
// is `Send`/`Sync`, and the generated code has no global state.
unsafe impl Send for CompiledSdpa {}
unsafe impl Sync for CompiledSdpa {}

impl CompiledSdpa {
    pub(crate) fn lowering_blobs(&self) -> Vec<Vec<u8>> {
        let pointers: Vec<*const u8> = if let Some(head) = self.head {
            vec![head as *const u8]
        } else {
            let (qk, softmax, sv) = self.split.expect("compiled split kernels");
            vec![qk as *const u8, softmax as *const u8, sv as *const u8]
        };
        pointers
            .into_iter()
            .zip(&self.kernel_lengths)
            .map(|(ptr, &len)| unsafe { std::slice::from_raw_parts(ptr, len).to_vec() })
            .collect()
    }

    /// Compiled specification.
    pub fn spec(&self) -> SdpaKernelSpec {
        self.spec
    }

    /// Machine code bytes produced.
    pub fn code_bytes(&self) -> usize {
        self.code_bytes
    }

    /// Required length of the scores buffer for `n` positions.
    ///
    /// The contract for the `-inf` padding (see `emit_sdpa`): the QK kernel
    /// writes up to the next multiple of 16. The fused shape, however, has no
    /// score buffer at all.
    pub fn scores_len(&self, n: usize) -> usize {
        if self.spec.form == SdpaForm::Fused {
            0
        } else {
            n.div_ceil(16) * 16
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    shape: SdpaShape,
    form: SdpaForm,
    exp: ExpStrategy,
    /// In the key, and not just in the spec: the two summation orders
    /// coexist within the same process (the block and the repeated decode
    /// compiled side by side). Without this field, the second requester
    /// would receive the kernel of the first, and a comparison of the two
    /// would compare a kernel with itself.
    qk_group: u8,
    /// In the key for **exactly the same reason as `qk_group`**, and the trap
    /// closed a second time: without this field, a comparison of the two
    /// branches compiled the “registers” branch, then received that same
    /// kernel for the “scratch” branch and concluded equality.
    sv_spill_all: bool,
}

type Cache = RwLock<HashMap<CacheKey, Option<Arc<CompiledSdpa>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Compile `spec`, or return the result already in cache.
///
/// `Ok(None)` = rejected form (or combination): the caller falls back to the
/// base path. `Err` = generator failure, which must propagate upward.
pub fn get_or_compile(spec: SdpaKernelSpec) -> Result<Option<Arc<CompiledSdpa>>> {
    let key = CacheKey {
        shape: spec.shape,
        form: spec.form,
        exp: spec.exp,
        qk_group: spec.qk_group,
        sv_spill_all: spec.sv_spill_all,
    };
    if let Some(hit) = cache().read().unwrap().get(&key) {
        return Ok(hit.clone());
    }
    let (built, why) = match compile(spec) {
        Ok(c) => (Some(Arc::new(c)), None),
        Err(CompileSdpaError::Declined(d)) => (None, Some(d)),
        Err(CompileSdpaError::Failed(e)) => return Err(e),
    };
    dump_plan(spec, built.as_deref(), why);
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(key).or_insert(built).clone())
}

/// The SDPA kernel of the process for this shape, per the installed knobs.
/// `Ok(None)` = the shape is declined.
pub fn head_kernel(shape: SdpaShape) -> Result<Option<Arc<CompiledSdpa>>> {
    // Two ISAs. Without AVX-512, the AVX2 kernel is used — but it serves
    // only one form and one exp strategy, and it does not perform QK in
    // groups (see `emit_sdpa_avx2`, which explains why). The knobs for form,
    // exp and group therefore do not apply to it: honoring
    // them would produce a refusal instead of a kernel, for a reason unrelated
    // to the machine.
    let avx2_only = !is_supported();
    #[cfg(target_arch = "x86_64")]
    if avx2_only && !crate::jit::emit_sdpa_avx2::is_supported() {
        return Ok(None);
    }
    #[cfg(not(target_arch = "x86_64"))]
    if avx2_only {
        return Ok(None);
    }
    let spec = SdpaKernelSpec {
        shape,
        form: if avx2_only {
            SdpaForm::Chained
        } else {
            sdpa_form()
        },
        exp: if avx2_only {
            ExpStrategy::Inline
        } else {
            sdpa_exp()
        },
        // The decoder keeps `4`: the scan of `pf_dist` covered the QK pass of
        // the **prefill block**, at `past` 1 024 and before `qk_pair`. Here the
        // stream loop streams K and V and the stride is bounded by the bus at 88
        // % — the knob exists so that this scan can be run.
        pf_dist: sdpa_pf_dist().unwrap_or(crate::jit::emit_sdpa::SDPA_PF_DIST),
        // The **same** knob as the prefill, and that is the point: bit-for-
        // bit equality between prefill and decode only holds if both ends
        // change order together. `qk_group_for` only touches it for the fused
        // form, which has no corresponding prefill block.
        qk_group: if avx2_only {
            0
        } else {
            qk_group_for(sdpa_form(), sdpa_qk_group(), qk_group_raw().is_some())
        },
        sv_spill_all: sdpa_sv_spill_all(),
    };
    match get_or_compile(spec)? {
        Some(k) => Ok(Some(k)),
        // A refused combination (`fused` with a shape it cannot hold: a sink, a QK group, a head
        // too wide for its registers) must not silently revert the entire model to scalar form: we
        // state it once, and we fall back to the chained form, which, in turn, serves all
        // geometries of the perimeter.
        None if spec.form == SdpaForm::Fused => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "WARN [backend_jit] sdpa shape=fused refused for head_dim={} \
                     (exp={}) — fallback to chained",
                    shape.head_dim,
                    spec.exp.name()
                );
            });
            get_or_compile(SdpaKernelSpec {
                form: SdpaForm::Chained,
                ..spec
            })
        }
        None => Ok(None),
    }
}

/// Compilation failure: assumed refusal or real crash.
#[derive(Debug)]
enum CompileSdpaError {
    Declined(Decline),
    Failed(HerbertError),
}

impl From<Decline> for CompileSdpaError {
    fn from(d: Decline) -> Self {
        CompileSdpaError::Declined(d)
    }
}

fn compile(spec: SdpaKernelSpec) -> std::result::Result<CompiledSdpa, CompileSdpaError> {
    // Without AVX-512, the AVX2 emitter is used. `spec.check()` carries
    // the domain of the AVX-512 kernel (shapes, 32-register budget, QK
    // groups); `check_avx2` carries that of the AVX2 kernel, which is narrower
    // and **named**. Applying the former here would accept a combination that
    // the AVX2 emitter would later reject — a late rejection, and at the wrong
    // name.
    #[cfg(target_arch = "x86_64")]
    if !is_supported() {
        crate::jit::emit_sdpa_avx2::check_avx2(&spec)?;
        let blob =
            crate::jit::emit_sdpa_avx2::emit_sdpa_avx2(&spec).map_err(CompileSdpaError::Failed)?;
        return finish(spec, vec![blob]);
    }
    spec.check()?;

    let blobs: Vec<Vec<u8>> = match spec.form {
        SdpaForm::Split => [Pass::Qk, Pass::Softmax, Pass::Sv]
            .iter()
            .map(|&p| emit_sdpa(&spec, Some(p)))
            .collect::<Result<Vec<_>>>()
            .map_err(CompileSdpaError::Failed)?,
        SdpaForm::Chained | SdpaForm::Fused => {
            vec![emit_sdpa(&spec, None).map_err(CompileSdpaError::Failed)?]
        }
    };
    finish(spec, blobs)
}

/// Places the blobs on an RX page and builds the [`CompiledSdpa`].
///
/// Extracted from [`compile`] so that both emitters — AVX-512 and AVX2 — share
/// **the same** layout, the same crash handler registration, and the same
/// pointer construction. A second copy of this code would have been derived.
fn finish(
    spec: SdpaKernelSpec,
    blobs: Vec<Vec<u8>>,
) -> std::result::Result<CompiledSdpa, CompileSdpaError> {
    let mut offsets = Vec::with_capacity(blobs.len());
    let mut total = 0usize;
    for blob in &blobs {
        offsets.push(total);
        total += blob.len().div_ceil(KERNEL_ALIGN) * KERNEL_ALIGN;
    }
    let exec = crate::jit::arch::map_exec(total.max(KERNEL_ALIGN), |map| {
        for (blob, &at) in blobs.iter().zip(offsets.iter()) {
            map[at..at + blob.len()].copy_from_slice(blob);
        }
    })
    .map_err(|e| {
        CompileSdpaError::Failed(HerbertError::Backend(format!("executable mapping: {e}")))
    })?;
    let lens: Vec<usize> = blobs.iter().map(|b| b.len()).collect();
    crate::jit::perf_map::register_pieces(exec.as_ptr(), &offsets, Some(&lens), total, |i| {
        format!("sdpa_{}#{i}", compact(&format!("{spec:?}")))
    });
    let base = exec.as_ptr();

    // SAFETY: each offset points to the start of a blob written above in a page
    // made executable, of which `exec` retains ownership. The signatures are
    // those that `emit_sdpa` emitted for this shape.
    let (head, split) = unsafe {
        match spec.form {
            SdpaForm::Split => (
                None,
                Some((
                    std::mem::transmute::<*const u8, SdpaQkFn>(base.add(offsets[0])),
                    std::mem::transmute::<*const u8, SdpaSoftmaxFn>(base.add(offsets[1])),
                    std::mem::transmute::<*const u8, SdpaSvFn>(base.add(offsets[2])),
                )),
            ),
            SdpaForm::Chained | SdpaForm::Fused => (
                Some(std::mem::transmute::<*const u8, SdpaHeadFn>(
                    base.add(offsets[0]),
                )),
                None,
            ),
        }
    };

    Ok(CompiledSdpa {
        spec,
        _exec: exec,
        head,
        split,
        kernel_lengths: blobs.iter().map(Vec::len).collect(),
        code_bytes: blobs.iter().map(|b| b.len()).sum(),
    })
}

/// Trace, once per shape, the kernel actually retained.
///
/// Same role as `dump_plan` of GEMM: a campaign CSV must be able to indicate on
/// which kernel it was produced, and a silent refusal must be readable.
fn dump_plan(spec: SdpaKernelSpec, built: Option<&CompiledSdpa>, why: Option<Decline>) {
    if !crate::jit::options::debug_plan() {
        return;
    }
    let head = format!(
        "sdpa head_dim={} sink={} form={} exp={} qk={} sv_acc={}/{} spill={} B",
        spec.shape.head_dim,
        spec.shape.sink,
        spec.form.name(),
        spec.exp.name(),
        spec.qk_group,
        spec.sv_acc_in_regs(),
        spec.shape.vecs(),
        64 * spec.sv_spill()
    );
    match built {
        Some(k) => eprintln!(
            "DEBUG [backend_jit] {head} bytes={} passes={}",
            k.code_bytes(),
            if k.head.is_some() { 1 } else { 3 }
        ),
        None => eprintln!(
            "DEBUG [backend_jit] {head} DECLINED ({})",
            why.map(|d| d.to_string()).unwrap_or_else(|| "?".into())
        ),
    }
}

/// A `Debug` form without spaces, for a symbol name.
fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}
