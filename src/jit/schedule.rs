//! The scheduler: the shape of a layer, its buffers, and the borrowed weights
//! it chains the generated kernels over.
//!
//! What it guarantees by construction:
//! - **no f32 in memory**: every buffer of [`Buffers`] is bf16, the only f32 are the SDPA scores
//!   and the GatedDeltaNet stages, internal to a layer;
//! - **no allocation during prefill or decode**: everything is sized once, in
//!   [`Scheduler::with_geometries`], from the model shape and a [`Budget`];
//! - **no conversion between two operations**: the producer writes what the consumer reads.
//!
//! Decode and prefill are the same lowering: what changes with the rows of a
//! block is the `M` of the GEMM (the weights are read once for `m` rows) and
//! the attention kernel (a block of queries, or one row). Causality costs
//! nothing: the query at position `p` sees `p + 1` keys.
//!
//! This file holds the types: [`ModelShape`], [`Budget`], [`Buffers`],
//! [`LayerWeights`], [`DeltaWeights`], [`Mixer`]; `scheduler.rs` compiles the
//! kernels at mount, `lowering.rs` turns a layer into nodes, `deltanet.rs`
//! compiles the kernels of the GatedDeltaNet stages.

use crate::core::error::{HerbertError, Result};
use crate::jit::elem::{get_or_compile as elem_kernel, CompiledElem};
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use crate::jit::ir::SdpaShape;
use crate::jit::recurrence::CompiledRecurrence;
use crate::jit::sdpa::{head_kernel, prefill_kernel, sdpa_rows, CompiledSdpa, CompiledSdpaPrefill};
use crate::jit::GemmShape;
use crate::pool::thread_pool::global_pool;
use std::sync::Arc;

/// Model geometry, fixed at load time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelShape {
    /// Residual stream dimension.
    pub hidden: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads (GQA).
    pub n_kv_heads: usize,
    /// Per-token dimension — the **maximum** across layers.
    ///
    /// Gemma4 carries two of them (256 on sliding layers, 512 on full ones): the
    /// `q`/`k`/`v`/`attn` buffers **and the KV stride** are sized to
    /// the largest, and each layer only uses the prefix that belongs to it. This
    /// is the pattern of [`Self::intermediate`], applied to the attention
    /// geometry; the **layer's** value is read from [`LayerWeights::head_dim`],
    /// and the resulting algebra in [`Self::for_layer`].
    pub head_dim: usize,
    /// Dimensions **rotated** by the RoPE — the model's **reference**
    /// value; the layer's value is [`LayerWeights::rotary_ndims`].
    ///
    /// `= head_dim` for a full RoPE; `< head_dim` for a **partial** RoPE
    /// (Qwen3.5: `partial_rotary_factor = 0.25` ⇒ 64 out of 256). The
    /// `head_dim − rotary_ndims` tail dimensions are **NoPE** and pass
    /// through unchanged.
    ///
    /// This is also the width of the tables: `cos`/`sin` carry
    /// `rotary_ndims/2` inputs per position, **not** `head_dim/2`.
    ///
    /// The pair convention is that of the reference implementation
    /// (`modeling_qwen3_5.py`, `rotate_half`): it
    /// applies **to the rotated subspace**, so the pair is
    /// `(j, j + rotary_ndims/2)` — and not `(j, j + head_dim/2)`, which is the
    /// Gemma-4 convention. Both exist in this repository; choosing one "by
    /// default" would make no sense.
    pub rotary_ndims: usize,
    /// Internal dimension of the MLP — the **maximum** over the layers.
    ///
    /// Gemma4-E2B supports two (6144 and 12288, `use_double_wide_mlp`): the
    /// `gate`/`up`/`h` buffers are sized for the largest, and each layer uses
    /// only its respective prefix. This is the only way to maintain "allocate
    /// once, zero allocation" with layers of different widths.
    pub intermediate: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Is the model norm in *Gemma* convention, `(1 + w)·rms(x)` ?
    ///
    /// `NormType::GemmaRMSNorm` of the config: Qwen3.5 stores every norm
    /// weight zero-centred.
    ///
    /// **Property of the entire model**, not just the Q/K norms: it applies
    /// to `input_layernorm`, `post_attention_layernorm` and the final norm.
    /// Hardcoding it only for Q/K would leave the others as `w·x`, without NaN
    /// or explosion — hence a shape field, which selects **all** norm kernels
    /// at once.
    pub norm_plus_one: bool,
    /// Explicit attention **scale** of the model, when it imposes one.
    ///
    /// `None` = the `1/√head_dim` convention. Gemma4 enforces one (it equals
    /// `query_pre_attn_scalar`, not `head_dim`), and hardcoding it yielded
    /// scores off by a constant factor — silently, since the following softmax
    /// remains normalized. Hence a field rather than a convention.
    pub attn_scale: Option<f32>,
    /// Does the model have an **attention output gate**? (Qwen3.5)
    ///
    /// `config.attn_output_gate`. When it is there, `q_proj` outputs `2·q_dim`:
    /// per head, `[head_dim of Q, head_dim of gate]` **interleaved**, and the
    /// SDPA output is weighted `attn ← attn · σ(gate)` before `o_proj`.
    ///
    /// **Entire model property**, like [`Self::norm_plus_one`]: it sizes
    /// the buffers, so it cannot vary from one layer to another without
    /// changing the reservation. An ungated layer of a gated model does not
    /// exist in the scope (in Qwen3.5 the only attention layers are the
    /// `full_attention` layers, all gated).
    pub output_gate: bool,
    /// Qwen2.5: `q_proj`, `k_proj`, `v_proj` have a bias, added to their outputs before the RoPE
    /// and the K/V append (the [`ElemOp::ResidualAdd`] kernel, the bias as the second operand
    /// at a row stride of zero).
    pub qkv_bias: bool,
    /// Gemma 4: the per-layer input branch and the logit soft cap; `None` on Qwen3.5.
    pub gemma: Option<GemmaShape>,
    /// `qwen3_5_moe`: the experts of its sparse layers; `None` on a dense model.
    pub qwen_moe: Option<QwenMoeShape>,
    /// Qwen3-VL: the layers after which an image block adds its deepstack vectors (3); 0
    /// elsewhere.
    pub deepstack_layers: usize,
}

/// The experts of a `qwen3_5_moe` model: the block of [`MoeShape`] and how the route weighs them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenMoeShape {
    /// The experts, their top-k and width.
    pub moe: MoeShape,
    /// `norm_topk_prob`: the `top_k` probabilities divided by their sum.
    pub norm_topk: bool,
    /// `lfm2_moe`: LiquidAI's sigmoid route with a bias per expert, and its
    /// `routed_scaling_factor`.
    pub lfm_scale: Option<f32>,
}

/// What the scheduler reserves and compiles for a Gemma 4 model (`model/gemma.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmaShape {
    /// Width of the per-layer input (`hidden_size_per_layer_input`).
    pub pl_dim: usize,
    /// Layers, so `n_layers · pl_dim` is the row of the per-layer inputs.
    pub n_layers: usize,
    /// `final_logit_softcapping`, 0 = none.
    pub softcap: f32,
    /// Sliding-window layers use at most this many positions; the largest value over the layers (0
    /// = no layer).
    pub window: usize,
    /// The mixture-of-experts block of every layer (the 26B-A4B); the decode works one row at a
    /// time, the x86-64 prefill groups the rows of a block by expert.
    pub moe: Option<MoeShape>,
}

/// The mixture-of-experts block a model reserves buffers for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeShape {
    /// Experts of a layer.
    pub experts: usize,
    /// Experts a token uses.
    pub top_k: usize,
    /// Width of an expert.
    pub inter: usize,
}

// The attention output gate (`config.attn_output_gate`, Qwen3.5) is fused in
// `q_proj`: the stored matrix interleaves per head, `[Q₀ g₀ Q₁ g₁ …]`, and one
// GEMM writes `2·q_dim` per row. Nothing is ever separated or copied: the
// loader permutes the rows of `wq` once, at load (`wq_source_row`,
// `permute_wq_rows`), so that the GEMM writes `[Q₀ Q₁ … | g₀ g₁ …]`, `Q`
// compact at the head of each row and the gate compact right after it, in
// the same `2·q_dim` buffer. Downstream, `Q` is read at a head stride of
// `head_dim` and a row stride of `2·q_dim` (`ModelShape::q_row_stride`), and
// the gate of row `r` is `q[r·2·q_dim + q_dim ..]`, applied with the
// whole-row `SigmoidMul` kernel. A row permutation permutes the outputs and
// nothing else; measured bit-exact against the reference on both executors.

/// The source row of row `n` of the permuted `wq`.
///
/// The stored `q_proj` interleaves per head: row `h·2·hd + j` is `Q` of head
/// `h`, row `h·2·hd + hd + j` its gate. The permuted matrix puts all of `Q`
/// first: row `n = h·hd + j` (for `n < q_dim`) reads `h·2·hd + j`, and row
/// `n = q_dim + h·hd + j` reads `h·2·hd + hd + j`.
pub fn wq_source_row(n: usize, n_heads: usize, head_dim: usize) -> usize {
    let q_dim = n_heads * head_dim;
    debug_assert!(n < 2 * q_dim);
    let (half, m) = if n < q_dim {
        (0, n)
    } else {
        (head_dim, n - q_dim)
    };
    let (h, j) = (m / head_dim, m % head_dim);
    h * 2 * head_dim + half + j
}

/// The rows of a row-major `[2·q_dim, k]` matrix reordered by
/// [`wq_source_row`]. Refuses (returns `None`) a matrix whose length is not
/// `2·n_heads·head_dim·k`.
pub fn permute_wq_rows<T: Copy>(
    src: &[T],
    n_heads: usize,
    head_dim: usize,
    k: usize,
) -> Option<Vec<T>> {
    let rows = 2 * n_heads * head_dim;
    if rows == 0 || k == 0 || src.len() != rows.checked_mul(k)? {
        return None;
    }
    let mut out = Vec::with_capacity(src.len());
    for n in 0..rows {
        let from = wq_source_row(n, n_heads, head_dim) * k;
        out.extend_from_slice(&src[from..from + k]);
    }
    Some(out)
}

impl ModelShape {
    /// The mixture-of-experts block the buffers are reserved for: Gemma 4's or Qwen's.
    pub fn experts(&self) -> Option<MoeShape> {
        self.gemma
            .and_then(|g| g.moe)
            .or(self.qwen_moe.map(|q| q.moe))
    }

    /// Dimension of the `q` projection (= `n_heads · head_dim`).
    pub const fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Table `cos`/`sin` input entries by position (= `rotary_ndims/2`).
    ///
    /// There is only one place where this division is written: the table width,
    /// the read stride per line, and the compiled kernel dimension must align,
    /// and writing them three times is exactly how a partial RoPE silently
    /// calculates falsely.
    pub const fn rope_half(&self) -> usize {
        self.rotary_ndims / 2
    }

    /// Width **written by the GEMM `q`**: `2·q_dim` when `q_proj` carries the
    /// fused gate, `q_dim` otherwise.
    pub fn q_gemm_dim(&self) -> usize {
        if self.output_gate {
            2 * self.q_dim()
        } else {
            self.q_dim()
        }
    }

    /// Distance between two **rows** in the `q` buffer: what the GEMM
    /// writes, `2·q_dim` on a gated model (`Q` compact, the gate compact
    /// after it, on the same row), `q_dim` without a gate. The head stride
    /// in `q` is always `head_dim`.
    pub(crate) fn q_row_stride(&self) -> usize {
        self.q_gemm_dim()
    }

    /// The actual scale passed to the SDPA kernel.
    pub fn scale(&self) -> f32 {
        self.attn_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }

    /// Dimensions of `k` and `v` projections.
    pub const fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Query heads per kv head.
    pub const fn gqa(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// The shape as **the layer** sees it: `head_dim` and `rotary_ndims` come
    /// from it, the rest of the model.
    ///
    /// This is the **only** point where a layer's algebra is derived:
    /// [`Self::q_dim`], [`Self::kv_dim`], [`Self::q_gemm_dim`], the two steps of
    /// `q`, [`Self::rope_half`] and [`Self::scale`] are recalculated by the
    /// **same** functions, based on the layer's value. Rewriting `nh · hd` by
    /// hand at each of these sites is exactly how a `head_dim` per layer
    /// calculates falsely in silence — and [`Self::scale`] is the most
    /// silent of the six (`1/√head_dim` by default: a factor 1.41 that softmax
    /// renormalizes, neither NaN nor out-of-bounds).
    ///
    /// `n_kv_heads` follows the same regime as `head_dim`: [`Self::n_kv_heads`]
    /// is merely the **reserved maximum** (Gemma4-26B: 8 out of 25 sliding
    /// layers, 2 out of 5 full layers), and it is [`Self::kv_dim`] — the width
    /// of the GEMMs `k`/`v` — and [`Self::gqa`] — the grouping passed to the
    /// SDPA kernel — that derive from it. Both calculate incorrectly in silence
    /// if given the model's maximum on a layer that has fewer: the GEMM would
    /// write `8·hd` columns where the weights only carry `2·hd`.
    pub fn for_layer(&self, head_dim: usize, rotary_ndims: usize, n_kv_heads: usize) -> Self {
        Self {
            head_dim,
            rotary_ndims,
            n_kv_heads,
            ..*self
        }
    }
}

/// What the caller reserves, once, at prompt launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Budget {
    /// Reserved KV positions. Exact bound of the scores buffer: it is the only
    /// buffer whose size depends on the context.
    pub kv_reserve_tokens: usize,
    /// **Reserved** lines for a prefill pass. `1` = decode only.
    ///
    /// This is a **ceiling**, not an imposed size: a pass processes the `m`
    /// given to it, and a longer prompt is split into chunks. Reserving
    /// generously guarantees that no allocation will occur afterward,
    /// regardless of the prompt length — that's all it does.
    ///
    /// This is the **only** dimension that causes activation buffers to grow:
    /// they weigh `max_batch ·` (the footprint of a row), 36 KiB per row on
    /// Qwen3-0.6B.
    pub max_batch: usize,
}

/// Rows **reserved** for the prefill.
///
/// This is a **bound, not a pass size**: a 300-token prompt passes in a single
/// 300-pass (the runner splits prompts into `chunks(max_batch)`).
/// Reserving generously therefore costs nothing in work — that is precisely the
/// point of the decision: **a reservation that covers all prompts in the
/// regime, so that no allocation ever needs to occur subsequently.**
///
/// Measured on a 12-core Zen 5:
/// at pp=1024, 1 → 44 tok/s, 128 → 2 108, 256 → 2 722, 1024 → 2 905. Most of
/// the gain is between 1 and 128; beyond that the curve is flat within the
/// noise.
///
/// Replayed with alternated arms (N=7, median), 2048 rows is the most
/// stable arm (±0,2 %) and the fastest.
///
/// What has been unfused: nothing forces to **process** by 2048 what is
/// **reserved** for 2048. This is now the role of [`prefill_pass`] ; this
/// constant is now just a **reservation ceiling**.
pub const DEFAULT_MAX_BATCH: usize = 2048;

/// Lines processed **in one go** during prefill — to be
/// distinguished from [`DEFAULT_MAX_BATCH`], which only
/// specifies what is **reserved**.
///
/// This is the "unbinding" that the note above referred to: the reservation
/// remains wide (no subsequent allocation, regardless of the prompt), and
/// the pass is small (the f32 buffers stay in the L3). Splitting changes
/// **no value** — the pieces already follow one another by construction.
///
/// Default **256**, over two measurements and one rule:
/// - `dn_qkv` weighs `m · 6144 · 4` bytes, i.e., **48.00 MiB at 2048 rows** — beyond the 32 MiB L3,
///   hence a **violation** of the format rule ; **6.00 MiB at 256**, which fits. `dn_z` and
///   `dn_out` follow (16 → 2 MiB).
/// - the speed **opposes it a bit, and we know by how much**: splitting costs **−1.4%** of prefill
///   on Qwen3.5-2B and **−0.7%** on the 4B, bars disjoint at ±0.2% on N=7 alternated, decode
///   intact, **identical text** in greedy. The invariant decides, and its price is measured instead
///   of assumed.
///
/// The value in force is [`crate::jit::knobs::Knobs::prefill_pass`].
pub fn prefill_pass() -> usize {
    crate::jit::knobs::knobs().prefill_pass
}

/// The default for [`prefill_pass`].
pub const PREFILL_PASS_DEFAULT: usize = 256;

// The pass is **bounded** by the reservation (`.min(max_batch)` where the mounted model lowers its
// prefill regime, `model/mounted.rs`): a default value exceeding the ceiling would not be rejected,
// it would be **silently clamped** to the ceiling, and the format rule would fall back without any
// indication. Verify this at compile time.
const _: () = assert!(PREFILL_PASS_DEFAULT <= DEFAULT_MAX_BATCH);

/// The maxima **over the layers** that size the GatedDeltaNet mixer buffers
/// — same role as [`ModelShape::intermediate`], which is also only the
/// reserved maximum.
///
/// The geometry **of the layer** is read from [`DeltaWeights`], never here:
/// this is only used to reserve, like [`Budget`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeltaShape {
    /// Maximum Q/K head width, used to compile the L2 body at mount.
    pub head_k_dim: usize,
    /// `2·key_dim + value_dim` — 6144 on Qwen3.5.
    pub conv_dim: usize,
    /// 2048 on Qwen3.5.
    pub value_dim: usize,
    /// 16 (2B) or 32 (4B). **Multiple of 16 required** by the R16 GEMM.
    pub num_v_heads: usize,
    /// `K` of the causal conv — 4.
    pub conv_kernel: usize,
}

/// All buffers of the layer, **in bf16**, allocated once.
///
/// Each buffer holds `max_batch` contiguous lines — line `r` of a buffer of
/// width `d` starts at `r · d`. On Qwen3-0.6B a line weighs ≈ 38 KiB,
/// excluding the scores buffer.
pub struct Buffers {
    /// Residual flux — the input and output of the layer.
    pub x: crate::core::avec::AVec<u16>,
    /// RMSNorm output.
    pub x_norm: crate::core::avec::AVec<u16>,
    /// Projection `q`.
    pub q: crate::core::avec::AVec<u16>,
    /// Projection `k` of the current positions.
    pub k: crate::core::avec::AVec<u16>,
    /// Projection of the current positions `v`.
    pub v: crate::core::avec::AVec<u16>,
    /// SDPA output, before `o_proj`.
    ///
    /// The attention output gate has no buffer of its own: it is read in
    /// place from `q` in both layouts.
    pub attn: crate::core::avec::AVec<u16>,
    /// Output of `o_proj`, then of `down_proj`.
    pub proj_out: crate::core::avec::AVec<u16>,
    /// Projection `gate` of the MLP.
    /// SwiGLU overwrites it in place before `down_proj` consumes it.
    pub gate: crate::core::avec::AVec<u16>,
    /// MLP `up` projection.
    pub up: crate::core::avec::AVec<u16>,
    /// SDPA scores — the only f32 **on the attention path**, and it does not exit
    /// the kernel.
    ///
    /// (The GatedDeltaNet mixer, for its part, works in f32 end-to-end: see
    /// the `dn_*` buffers below, reserved by [`Scheduler::with_geometries_delta`]
    /// only.)
    ///
    /// One `scores_stride` slot per worker: during prefill, the blocks (head,
    /// row chunk) run in parallel and each needs its own. A single slot holds
    /// the `rows` lines of a block, not just one.
    pub scores: crate::core::avec::AVec<f32>,
    /// GatedDeltaNet: output of `Wqkv`, `max_batch · conv_dim` f32.
    ///
    /// **Empty** without a delta geometry: the `dn_*` buffers are only
    /// reserved by [`Scheduler::with_geometries_delta`]. In f32 because the
    /// conv, gates and recurrence work in f32 — going through the GEMM with a
    /// bf16 output would lose, silently, precision the reference keeps.
    pub dn_qkv: crate::core::avec::AVec<f32>,
    /// GatedDeltaNet : output of `Wz`, `max_batch · value_dim` f32.
    pub dn_z: crate::core::avec::AVec<f32>,
    /// GatedDeltaNet : outputs of `Wa` and `Wb`, `max_batch · 2 · num_v_heads`
    /// f32.
    ///
    /// Fixed disposition: line `r` occupies `[r·2·nv, (r+1)·2·nv)`, with `a` on
    /// `[0, nv)` and `b` on `[nv, 2·nv)`.
    pub dn_ab: crate::core::avec::AVec<f32>,
    /// GatedDeltaNet : output of the recurrence followed by the gated
    /// norm, `max_batch · value_dim` f32.
    pub dn_out: crate::core::avec::AVec<f32>,
    /// GatedDeltaNet: `dn_out` narrowed to bf16, `max_batch · value_dim` u16 —
    /// **the input of `Wo`**.
    ///
    /// `proj()` requires bf16 and the recurrence produces f32: this buffer is
    /// the boundary, and it is the **only** one in the entire arm. This is the
    /// engine's convention — the attention path also passes through a bf16
    /// `attn` before `wo`.
    ///
    /// An assumed deviation from the reference, which keeps f32 until
    /// `out_proj`: the criterion is the greedy output, not the bit.
    pub dn_out_bf16: crate::core::avec::AVec<u16>,
    /// GatedDeltaNet: the gate `beta = σ(b)`, `max_batch · num_v_heads` f32.
    ///
    /// Sized to `max_batch` rows and not just to the `num_v_heads` of a
    /// token: compute the gates of the `m` rows at once, taking the
    /// `exp`/`softplus` out of the sequential recurrence, for up to 512 KiB
    /// at worst. Line `r` on `[r·nv, (r+1)·nv)`.
    pub dn_beta: crate::core::avec::AVec<f32>,
    /// GatedDeltaNet: the gate `decay = exp(−exp(a_log)·dt)`,
    /// `max_batch · num_v_heads` f32. Same layout as `dn_beta`.
    pub dn_decay: crate::core::avec::AVec<f32>,
    /// The activations of the current projections packed for the target's
    /// matrix engine ([`crate::jit::emit::packed_a_bytes`]), rewritten before
    /// each group of projections sharing an input. Empty when the target's
    /// GEMM plan never asks for it.
    pub packed: crate::core::avec::AVec<u16>,
    /// Gemma 4: the per-layer inputs of the rows in flight, `max_batch · layers · pl_dim`, ready
    /// for the layers (`lower_input`); empty on Qwen3.5.
    pub ple_fold: crate::core::avec::AVec<u16>,
    /// Gemma 4: scratch of the same size (lookup, then projection).
    pub ple_tmp: crate::core::avec::AVec<u16>,
    /// Gemma 4: the `pl_dim`-wide gate of a layer.
    pub ple_gate: crate::core::avec::AVec<u16>,
    /// Gemma 4 MoE (`max_batch` rows): the residual normed for the router, the router scores
    /// (bf16), the dense branch, the residual normed for the experts, the `top_k` rows of
    /// gate/up, of activated, of down outputs, the `top_k` weights (f32), the experts' sum.
    pub moe_router_in: crate::core::avec::AVec<u16>,
    /// The router scores, bf16, one per expert.
    pub moe_scores: crate::core::avec::AVec<u16>,
    /// The dense branch of the layer, normed.
    pub moe_dense: crate::core::avec::AVec<u16>,
    /// The residual normed for the experts.
    pub moe_expert_in: crate::core::avec::AVec<u16>,
    /// `top_k` rows of `2 · inter`: gate then up.
    pub moe_gate_up: crate::core::avec::AVec<u16>,
    /// `top_k` rows of `inter`.
    pub moe_act: crate::core::avec::AVec<u16>,
    /// `top_k` rows of `hidden`.
    pub moe_down: crate::core::avec::AVec<u16>,
    /// The `top_k` weights of the chosen experts, f32.
    pub moe_weights: crate::core::avec::AVec<f32>,
    /// The sum of the experts, bf16.
    pub moe_sum: crate::core::avec::AVec<u16>,
    /// A block of rows (the prefill): the place of each `(row, k)` among the rows gathered by
    /// expert, u32.
    pub moe_pos: crate::core::avec::AVec<u32>,
    /// The expert of each `(row, k)`, u32.
    pub moe_ids: crate::core::avec::AVec<u32>,
    /// The quantized expert input rows gathered by expert (`max_batch · top_k` rows, bytes as
    /// u16).
    pub moe_gathered: crate::core::avec::AVec<u16>,
    /// The quantized activations of the down projections, same order (bytes as u16).
    pub moe_act_q: crate::core::avec::AVec<u16>,
}

impl Buffers {
    /// Allocates everything, once, from the shape and budget.
    fn new(shape: ModelShape, budget: Budget, scores_stride: usize, slots: usize) -> Self {
        let m = budget.max_batch.max(1);
        // `q` carries the raw output of the GEMM, so `2·q_dim` when the gate is
        // fused there; the gate is read in place.
        let q_len = m * shape.q_row_stride();
        // Gemma 4 only: the router input, the dense branch, the expert input and the sum (Qwen
        // reads `x_norm` and adds into `x` directly).
        let moe_rows = |len: usize| {
            shape
                .gemma
                .and_then(|g| g.moe)
                .map_or_else(crate::core::avec::AVec::new, |_| {
                    crate::core::avec::AVec::zeroed(len)
                })
        };
        let moe = shape.experts();
        // The buffers of a block of rows only: the decode (one row) never reads them.
        fn moe_block<T: crate::core::avec::Zeroable>(
            moe: Option<MoeShape>,
            m: usize,
            len: impl Fn(MoeShape) -> usize,
        ) -> crate::core::avec::AVec<T> {
            match moe {
                Some(e) if m > 1 => crate::core::avec::AVec::zeroed(len(e)),
                _ => crate::core::avec::AVec::new(),
            }
        }
        Self {
            x: crate::core::avec::AVec::zeroed(m * shape.hidden),
            x_norm: crate::core::avec::AVec::zeroed(m * shape.hidden),
            q: crate::core::avec::AVec::zeroed(q_len),
            k: crate::core::avec::AVec::zeroed(m * shape.kv_dim()),
            v: crate::core::avec::AVec::zeroed(m * shape.kv_dim()),
            attn: crate::core::avec::AVec::zeroed(m * shape.q_dim()),
            proj_out: crate::core::avec::AVec::zeroed(m * shape.hidden),
            gate: crate::core::avec::AVec::zeroed(m * shape.intermediate),
            up: crate::core::avec::AVec::zeroed(m * shape.intermediate),
            scores: crate::core::avec::AVec::zeroed(scores_stride * slots),
            // Reserved by `Scheduler::with_geometries_delta`, which replaces them
            // afterwards: a model without a linear mixer pays nothing.
            dn_qkv: crate::core::avec::AVec::new(),
            dn_z: crate::core::avec::AVec::new(),
            dn_ab: crate::core::avec::AVec::new(),
            dn_out: crate::core::avec::AVec::new(),
            dn_out_bf16: crate::core::avec::AVec::new(),
            dn_beta: crate::core::avec::AVec::new(),
            dn_decay: crate::core::avec::AVec::new(),
            packed: crate::core::avec::AVec::new(),
            ple_fold: shape.gemma.map_or_else(crate::core::avec::AVec::new, |g| {
                crate::core::avec::AVec::zeroed(m * g.n_layers * g.pl_dim)
            }),
            ple_tmp: shape.gemma.map_or_else(crate::core::avec::AVec::new, |g| {
                crate::core::avec::AVec::zeroed(m * g.n_layers * g.pl_dim)
            }),
            ple_gate: shape.gemma.map_or_else(crate::core::avec::AVec::new, |g| {
                crate::core::avec::AVec::zeroed(m * g.pl_dim)
            }),
            moe_router_in: moe_rows(m * shape.hidden),
            moe_scores: moe.map_or_else(crate::core::avec::AVec::new, |e| {
                crate::core::avec::AVec::zeroed(m * e.experts)
            }),
            moe_dense: moe_rows(m * shape.hidden),
            moe_expert_in: moe_rows(m * shape.hidden),
            moe_gate_up: moe.map_or_else(crate::core::avec::AVec::new, |e| {
                crate::core::avec::AVec::zeroed(m * e.top_k * 2 * e.inter)
            }),
            moe_act: moe.map_or_else(crate::core::avec::AVec::new, |e| {
                crate::core::avec::AVec::zeroed(m * e.top_k * e.inter)
            }),
            moe_down: moe.map_or_else(crate::core::avec::AVec::new, |e| {
                crate::core::avec::AVec::zeroed(m * e.top_k * shape.hidden)
            }),
            moe_weights: moe.map_or_else(crate::core::avec::AVec::new, |e| {
                crate::core::avec::AVec::zeroed(m * e.top_k.max(8))
            }),
            moe_sum: moe_rows(m * shape.hidden),
            moe_pos: moe_block(moe, m, |e| m * e.top_k),
            moe_ids: moe_block(moe, m, |e| m * e.top_k),
            moe_gathered: moe_block(moe, m, |e| {
                crate::jit::int8::a_bytes(m * e.top_k, shape.hidden).div_ceil(2)
            }),
            moe_act_q: moe_block(moe, m, |e| {
                crate::jit::int8::a_bytes(m * e.top_k, e.inter).div_ceil(2)
            }),
        }
    }

    /// Actually reserved bytes — what we can announce at launch.
    pub fn bytes(&self) -> usize {
        2 * (self.x.len()
            + self.x_norm.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn.len()
            + self.proj_out.len()
            + self.gate.len()
            + self.up.len()
            + self.dn_out_bf16.len()
            + self.packed.len()
            + self.ple_fold.len()
            + self.ple_tmp.len()
            + self.ple_gate.len()
            + self.moe_router_in.len()
            + self.moe_scores.len()
            + self.moe_dense.len()
            + self.moe_expert_in.len()
            + self.moe_gate_up.len()
            + self.moe_act.len()
            + self.moe_down.len()
            + self.moe_sum.len()
            + self.moe_gathered.len()
            + self.moe_act_q.len()
            + 2 * (self.moe_pos.len() + self.moe_ids.len()))
            + 4 * (self.scores.len()
                + self.dn_qkv.len()
                + self.dn_z.len()
                + self.dn_ab.len()
                + self.dn_out.len()
                + self.dn_beta.len()
                + self.dn_decay.len())
    }
}

/// The MLP activation of a layer.
///
/// The generator emits both under the same ABI: `geluTanh` is a SwiGLU whose
/// sigmoid argument is a cubic (`emit_elem.rs`, `ElemOp::GeluTanh`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    /// `g · σ(g) · u` — Qwen3, Mistral, Llama.
    SwiGlu,
    /// `½·g·(1 + tanh(z)) · u` — Gemma4.
    GeluTanh,
}

/// The weights of a GatedDeltaNet mixer, borrowed.
///
/// Everything is **bf16 at the checkpoint** except `a_log`, the only true f32
/// and the most sensitive
/// (`decay = exp(−exp(a_log)·dt)`, applied to the state at each step).
pub struct DeltaWeights<'a> {
    /// `[conv_dim, hidden]`, R16. `conv_dim = key_dim + key_dim + value_dim`.
    pub in_proj_qkv: &'a [u16],
    /// `[value_dim, hidden]`, R16.
    pub in_proj_z: &'a [u16],
    /// `[num_v_heads, hidden]`, R16.
    pub in_proj_a: &'a [u16],
    /// `[num_v_heads, hidden]`, R16.
    pub in_proj_b: &'a [u16],
    /// `[conv_dim, conv_kernel]`, bf16 **flat** — `w[ch·conv_kernel + j]`.
    /// **No bias** (`bias=False`).
    ///
    /// Not R16: that layout is for the GEMM, and a depthwise conv does not
    /// go through it.
    pub conv1d_weight: &'a [u16],
    /// `[num_v_heads]` — **the only f32**.
    pub a_log: &'a [f32],
    /// `[num_v_heads]`, R16.
    pub dt_bias: &'a [u16],
    /// `[head_v_dim]`, bf16 — the gated norm. F32 at the checkpoint, rounded
    /// to bf16 at load like the reference implementation does.
    pub norm_weight: &'a [u16],
    /// `[hidden, value_dim]`, R16.
    pub out_proj: &'a [u16],
    /// Linear mixer key heads.
    pub num_k_heads: usize,
    /// Value heads. In 4B they are **twice** as numerous as the key heads
    /// (`hpg = num_v_heads / num_k_heads = 2`).
    pub num_v_heads: usize,
    /// Dimension of a key head (128 across the entire Qwen3.5 family).
    pub head_k_dim: usize,
    /// Value head dimension (128, same).
    pub head_v_dim: usize,
    /// Length of the causal conv1d kernel (`K = 4`).
    pub conv_kernel: usize,
    /// Granite 4's Mamba2 in the DeltaNet's places (see `model::GatedDeltaNet::mamba`): no L2 of q
    /// and k, the step `dt = softplus(a + dt_bias)` in place of `beta`, the update `s ←
    /// decay·s + (dt·B) ⊗ x` without the delta rule, `y += D·x`, then `w · rms(y · silu(z))`
    /// over the whole width.
    pub mamba: bool,
    /// `[conv_dim]` f32 (Mamba2); empty otherwise.
    pub conv1d_bias: &'a [f32],
    /// `[num_v_heads]` f32 (Mamba2); empty otherwise.
    pub d: &'a [f32],
    /// LiquidAI's gated short convolution in the DeltaNet's places (see
    /// `model::GatedDeltaNet::short_conv`): `in_proj_z` gives `B`, `in_proj_c` `C`,
    /// `in_proj_qkv` `x`, `[hidden, hidden]` each; `conv1d_weight` tap-major by blocks of
    /// sixteen channels; `head_v_dim` the hidden width, no head.
    pub short_conv: bool,
    /// `[hidden, hidden]`, R16 (the short conv's `C`); empty otherwise.
    pub in_proj_c: &'a [u16],
}

/// What a Gemma 4 layer has beyond the dense layer, borrowed (`model/gemma.rs`). The norm in front
/// of the MLP is [`LayerWeights::post_attn_norm`].
pub struct GemmaLayerWeights<'a> {
    /// `post_attention_layernorm`: norm of the attention output before the residual.
    pub post_attn_out_norm: &'a [u16],
    /// `post_feedforward_layernorm`: norm of the MLP output before the residual.
    pub post_ff_norm: &'a [u16],
    /// Positions a sliding layer sees, 0 = full attention.
    pub window: usize,
    /// Layer whose cache this layer reads; `None` = its own.
    pub kv_source: Option<usize>,
    /// The per-layer input branch; `None` on a model without per-layer inputs.
    pub ple: Option<PleLayerWeights<'a>>,
    /// No `v_proj`: the values are the keys, before the key norm and the RoPE.
    pub k_eq_v: bool,
    /// The mixture-of-experts block; `None` on a dense layer.
    pub moe: Option<MoeLayerWeights<'a>>,
    /// Multiplier of the residual at the end of the layer.
    pub layer_scalar: f32,
}

/// The mixture-of-experts block of a Gemma 4 layer, borrowed.
pub struct MoeLayerWeights<'a> {
    /// The router in blocks of sixteen experts, R16 bf16 each.
    pub router: Vec<&'a [u16]>,
    /// The weight of the norm in front of the router.
    pub router_scale: &'a [u16],
    /// `router.per_expert_scale`, f32.
    pub per_expert_scale: &'a [f32],
    /// Norm of the residual in front of the experts.
    pub pre_ff_norm_2: &'a [u16],
    /// Norm of the dense branch.
    pub post_ff_norm_1: &'a [u16],
    /// Norm of the sum of the experts.
    pub post_ff_norm_2: &'a [u16],
    /// Per expert, R16: `[2 · inter, hidden]` and `[hidden, inter]`.
    pub gate_up: Vec<&'a [u16]>,
    /// Per expert, R16: `[hidden, inter]`.
    pub down: Vec<&'a [u16]>,
}

/// The per-layer input branch of a Gemma 4 layer, borrowed.
pub struct PleLayerWeights<'a> {
    /// `[pl_dim, hidden]`, R16.
    pub gate: &'a [u16],
    /// `[hidden, pl_dim]`, R16.
    pub proj: &'a [u16],
    /// `[hidden]`.
    pub norm: &'a [u16],
}

/// The mixer of a layer. The per-layer variation lives in
/// [`LayerWeights`], **not** in [`ModelShape`] (`intermediate` /
/// `activation` pattern).
#[allow(clippy::large_enum_variant)]
pub enum Mixer<'a> {
    /// Dense attention.
    Attention,
    /// The linear mixer of Qwen3.5, with its weights.
    GatedDeltaNet(DeltaWeights<'a>),
}

/// The weights of a layer are borrowed. The projections are in R16 format,
/// the norms are bf16 vectors.
pub struct LayerWeights<'a> {
    /// Input RMSNorm weights.
    pub input_norm: &'a [u16],
    /// RMSNorm weights preceding the MLP — **the role, not the tensor**.
    ///
    /// On Qwen it is `post_attention_layernorm`; on Gemma 4 it is
    /// `pre_feedforward_layernorm`, with `post_attention_layernorm` then in
    /// [`GemmaLayerWeights::post_attn_out_norm`].
    pub post_attn_norm: &'a [u16],
    /// Norm weights per head on `q`.
    pub q_norm: &'a [u16],
    /// Weight of the norm per head on `k`.
    ///
    /// `q_norm` and `k_norm` empty: no per-head norm, the RoPE alone (Qwen2.5).
    pub k_norm: &'a [u16],
    /// The biases of `q`, `k`, `v` (Qwen2.5, [`ModelShape::qkv_bias`]); empty elsewhere.
    pub q_bias: &'a [u16],
    /// `[kv_dim]`.
    pub k_bias: &'a [u16],
    /// `[kv_dim]`.
    pub v_bias: &'a [u16],
    /// Projection `q`, format R16.
    ///
    /// On a gated model, it is the **fused** matrix `[Q; gate]` of `2·q_dim`
    /// rows, permuted at load ([`wq_source_row`]).
    pub wq: &'a [u16],
    /// Projection `k`, format R16.
    pub wk: &'a [u16],
    /// Projection `v`, format R16.
    pub wv: &'a [u16],
    /// Output projection, format R16.
    pub wo: &'a [u16],
    /// Projection `gate`, format R16.
    pub w_gate: &'a [u16],
    /// Projection `up`, format R16.
    pub w_up: &'a [u16],
    /// Projection `down`, format R16.
    pub w_down: &'a [u16],
    /// The MLP activation **of this layer**.
    pub activation: Activation,
    /// Internal dimension of the MLP **of this layer**.
    ///
    /// May differ from one layer to another (Gemma4-E2B: 6144 then 12288). The
    /// corresponding SwiGLU kernel must have been declared at
    /// [`Scheduler::with_geometries`].
    pub intermediate: usize,
    /// Number of KV heads **of this layer**, when it differs from the shape.
    ///
    /// `None` = that of [`ModelShape::n_kv_heads`], the case for the entire
    /// current fleet (all four models have it uniform). `Some(n)` on a model
    /// that varies it: Gemma4-26B-A4B carries 8 KV heads on its 25 sliding
    /// layers and 2 on its 5 full layers.
    ///
    /// [`ModelShape::n_kv_heads`] is then only the **reserved maximum** — the
    /// `k`/`v` buffers and the cache stride follow it, a layer uses only the
    /// prefix. This is the pattern of [`Self::head_dim`] and
    /// [`Self::intermediate`], applied to the third axis of attention.
    ///
    /// On a `Mixer::GatedDeltaNet` layer this field has no owner: nothing
    /// reads it. `None` is the safest choice.
    pub n_kv_heads: Option<usize>,
    /// Per-head dimension **of this layer**.
    ///
    /// May differ from one layer to another (Gemma4: 256 on sliding
    /// layers, 512 on full ones). [`ModelShape::head_dim`] is only the
    /// **reserved maximum**; the corresponding set of kernels must have been
    /// declared to [`Scheduler::with_geometries`].
    ///
    /// On a `Mixer::GatedDeltaNet` layer this field has no owner: nothing
    /// reads it. Putting the shape value there is the safest choice.
    pub head_dim: usize,
    /// Dimensions rotated by RoPE **for this layer**.
    ///
    /// **Distinct** axis from [`Self::head_dim`], and it cannot be derived from it: the
    /// partial RoPE of Qwen3.5 runs 64 out of 256, Gemma4 runs 512 out of 512.
    /// This value selects the RoPE **kernel** and the set of tables
    /// `cos`/`sin` (width `rotary_ndims/2`), not `head_dim`.
    pub rotary_ndims: usize,
    /// The mixer of **this layer**.
    ///
    /// In Qwen3.5, both coexist within the same model (18 `linear_attention`
    /// layers out of 24 in the 2B variant): the input norm, the MLP, and the
    /// two residuals are shared, **only the mixer body changes**.
    /// Hence a layer field, such as [`Self::intermediate`], and not a shape
    /// field.
    pub mixer: Mixer<'a>,
    /// Gemma 4 only.
    pub gemma: Option<GemmaLayerWeights<'a>>,
    /// `qwen3_5_moe` only, on a sparse layer: the experts; the dense MLP above is then the shared
    /// expert.
    pub qwen_moe: Option<QwenMoeLayerWeights<'a>>,
    /// LFM3 only.
    pub lfm3: Option<Lfm3LayerWeights<'a>>,
}

/// What an LFM3 layer (`lfm3`) has beyond LFM2's, borrowed: `x += post_mixer_norm(mixer(norm(x)))`,
/// `x += post_mlp_norm(mlp(norm(x)))`; on its attention layers each head's output gated by
/// `bf16(σ(g_h))`, `g = head_gate · norm(x)`, and on a sliding layer the last `window` keys.
pub struct Lfm3LayerWeights<'a> {
    /// `post_operator_norm`, `[hidden]`; empty: no norm before the mixer's residual.
    pub post_mixer_norm: &'a [u16],
    /// `post_ffn_norm`, `[hidden]`; empty: no norm before the MLP's residual.
    pub post_mlp_norm: &'a [u16],
    /// `self_attn.gate_proj`, `[heads, hidden]`, R16 (bf16: sixteen rows stay off the int8 and
    /// 4-bit GEMMs); empty on a `conv` layer.
    pub head_gate: &'a [u16],
    /// The keys a sliding layer sees, itself included; 0 on a full layer. A sliding layer also
    /// reads its own RoPE tables (the whole head at its own base), mounted by the runner.
    pub window: usize,
}

/// The experts of a sparse `qwen3_5_moe` layer, borrowed.
pub struct QwenMoeLayerWeights<'a> {
    /// The router in blocks of sixteen experts, R16 bf16 each.
    pub router: Vec<&'a [u16]>,
    /// `mlp.shared_expert_gate`, `[hidden]` bf16.
    pub shared_gate: &'a [u16],
    /// Per expert, R16: `[2 · inter, hidden]` (gate rows then up rows).
    pub gate_up: Vec<&'a [u16]>,
    /// Per expert, R16: `[hidden, inter]`.
    pub down: Vec<&'a [u16]>,
    /// `lfm2_moe`: the bias per expert of the route's choice, f32; empty otherwise.
    pub expert_bias: &'a [f32],
}

/// The set of kernels for a single `head_dim` value.
///
/// Four kernels depend on this magnitude: `sdpa` and `sdpa_prefill` (whose
/// key is further the triplet `(head_dim, q_row_stride, q_dim)`), `norm_head`
/// on `head_dim`, `gate_row` on `n_heads · head_dim`. RoPE, however, does not depend
/// **on it**: its axis is `rotary_ndims` (see [`Scheduler::ropes`]).
///
/// The KV row stride **within** the kernel comes for free with this
/// partitioning: `head_dim` is an immediate (`ir.rs`, `kv_row_bytes`), so one
/// kernel per `head_dim` is also one KV access stride per `head_dim`.
struct HeadKernels {
    /// The value that indexes this set.
    head_dim: usize,
    norm_head: Arc<CompiledElem>,
    /// `attn ← attn · σ(gate)`, the whole row (`q_dim`) in one call, the gate
    /// read in place after `Q`. `None` when the model has no gate.
    gate_row: Option<Arc<CompiledElem>>,
    /// `x += bias` on the rows of `q` (`q_dim`) and of `k` and `v` (`kv_dim`); `None` without
    /// [`ModelShape::qkv_bias`].
    bias: Option<(Arc<CompiledElem>, Arc<CompiledElem>)>,
    sdpa: Arc<CompiledSdpa>,
    /// Core of prefill: `rows` queries against the same K/V. `None` when the
    /// budget allows only a single row (pure decode) or when the shape is
    /// rejected — in both cases the decode SDPA is called row by row, which it
    /// already did.
    sdpa_prefill: Option<Arc<CompiledSdpaPrefill>>,
}

/// The scheduler: compiled kernels and buffers, for a given shape.
pub struct Scheduler {
    shape: ModelShape,
    buffers: Buffers,
    /// Maximum number of lines that buffers can hold.
    max_batch: usize,
    /// Length of a scores location, in f32.
    scores_stride: usize,
    norm_hidden: Arc<CompiledElem>,
    /// A set of **distinct** kernels per `head_dim`.
    ///
    /// Linear scan, exactly like [`Self::swiglu`] and for the same reason: the
    /// target models have **one or two**, and a hash table would cost more
    /// than the scan — above all, it would allocate.
    heads: Vec<HeadKernels>,
    /// One activation kernel per distinct `(activation, width_of)` pair.
    ///
    /// Linear scan: target models have **one or two** (four shapes in
    /// Gemma4-E2B, but only two MLP widths, and a single activation per
    /// model). A hash table would be more expensive than the scan and would
    /// allocate.
    swiglu: Vec<(Activation, usize, Arc<CompiledElem>)>,
    residual: Arc<CompiledElem>,
    /// One RoPE kernel per `rotary_ndims` **distinct** — separate axis from
    /// `head_dim`, because the kernel only writes its first `spec.dim` u16s.
    ropes: Vec<(usize, Arc<CompiledElem>)>,
    /// The emitted **GatedDeltaNet recurrence** kernel, for the width `head_v_dim = value_dim /
    /// num_v_heads` of the [`DeltaShape`] — `None` without DeltaNet mixer, on a target without the
    /// emitter (x86 without AVX-512), or if the width exceeds the register budget, in which case
    /// the generated program refuses the mount. Compiled **at build time**.
    recurrence: Option<Arc<CompiledRecurrence>>,
    /// The emitted **causal conv + SiLU** kernel, for the `conv_kernel` of
    /// [`DeltaShape`] — `None` without a DeltaNet mixer, on x86 without AVX-512, or if `K ≠ 4`,
    /// in which case the generated program refuses the mount. Compiled **at mount time**.
    conv: Option<Arc<crate::jit::conv::CompiledConv>>,
    /// The emitted **gated RMS norm** kernel, for the
    /// `head_v_dim` of [`DeltaShape`] — `None` without DeltaNet mixer,
    /// on x86 without AVX-512, or if the width exits the shape (multiple of 16,
    /// ≤ 128), in which case the generated program refuses the mount. Compiled
    /// **at assembly**.
    gated_norm: Option<Arc<crate::jit::gated_norm::CompiledGatedNorm>>,
    /// Gemma 4: the extra kernels; `None` on Qwen3.5.
    gemma: Option<GemmaKernels>,
    /// bf16 ones, `shape.head_dim` long: the weight of the norm of the values (`v_norm` has none).
    ones: crate::core::avec::AVec<u16>,
    /// bf16 zeros, `max_batch · hidden` long, on a `qwen3_5_moe` or Granite 3 model: on a layer
    /// without a dense MLP (Granite 3), the output of the absent shared expert. Empty on a
    /// model without experts.
    zeros: crate::core::avec::AVec<u16>,
    /// Qwen3-VL: the deepstack vectors, `deepstack_layers · max_batch · hidden` bf16, layer after
    /// layer, which an image block adds to its rows after each of the first layers; zeros
    /// until the vision writes them. Empty elsewhere.
    deepstack: crate::core::avec::AVec<u16>,
}

/// The kernels a Gemma 4 model compiles beyond the dense ones.
struct GemmaKernels {
    /// `y = x · s` over a residual row (the embedding scale, the layer scalar).
    scale_hidden: Arc<CompiledElem>,
    /// The kernels of the per-layer inputs; `None` without them.
    ple: Option<PleKernels>,
}

struct PleKernels {
    /// RMSNorm over `pl_dim` (the per-layer inputs, one group each).
    norm_pl: Arc<CompiledElem>,
    /// `y = x · s` over the row of the per-layer inputs (`layers · pl_dim`).
    scale_ple: Arc<CompiledElem>,
    /// `a += b` over the row of the per-layer inputs.
    add_ple: Arc<CompiledElem>,
}

// ═════════════════════════════════════════════════════════════════════════════
// SUBMODULES
// ═════════════════════════════════════════════════════════════════════════════
mod scheduler;

mod pass;

mod deltanet;
pub(crate) use deltanet::{
    conv_at_compile_time, gated_norm_at_compile_time, recurrence_at_compile_time, DN_L2_EPS,
};
pub use deltanet::{spmd_attn_chain_team, spmd_elem_team};

pub(crate) mod lowering;

pub mod vision;
pub mod vision_qwen;

pub mod vision_lfm3;

pub mod audio;

pub mod encoder;
