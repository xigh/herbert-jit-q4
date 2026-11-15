//! What the loader builds: the model in memory, every matrix packed for the
//! emitted GEMM, every vector as the checkpoint stores it.
//!
//! For example, on Qwen3.5-4B: the embedding table `[248320, 2560]` bf16 (1.2 GiB),
//! 32 decoder layers (24 with a GatedDeltaNet mixer, 8 with attention), the
//! final norm, the language head (the embedding table packed R16 when tied),
//! and the RoPE tables for `max_position_embeddings` positions.

use crate::core::config::Config;
use crate::core::tensor::BF16;
use crate::model::weight::PackedBf16R16Weight;

pub struct Model {
    /// `[vocab, hidden]`, row-major bf16: one row is copied per token.
    pub embed_tokens: crate::core::avec::AVec<BF16>,
    pub decoder_layers: Vec<DecoderLayer>,
    /// The final RMSNorm weight, `[hidden]`.
    pub norm: crate::core::avec::AVec<BF16>,
    /// `[vocab, hidden]` packed R16: the last GEMM of a token, `N = vocab`.
    pub lm_head: PackedBf16R16Weight,
    pub config: Config,
    /// `cos(pos · θ_i)`, `[max_position_embeddings, rotary_ndims / 2]` f32.
    pub cos_cache: crate::core::avec::AVec<f32>,
    /// `sin`, same layout.
    pub sin_cache: crate::core::avec::AVec<f32>,
    /// Gemma 4 only: the per-layer embeddings and the second RoPE table.
    pub gemma: Option<crate::model::gemma::GemmaModel>,
    /// SmolLM3: the tables of the layers without position, `[positions, NOPE_ROTARY_NDIMS / 2]`,
    /// cos 1 and sin 0.
    pub rope_nope: Option<(crate::core::avec::AVec<f32>, crate::core::avec::AVec<f32>)>,
    /// LFM3: the tables of the sliding layers (`rope_parameters.sliding_attention`, the whole head
    /// rotated), `[positions, rotary_ndims / 2]`; `cos_cache` and `sin_cache` are then the
    /// full layers' zero-tail tables.
    pub rope_sliding: Option<(crate::core::avec::AVec<f32>, crate::core::avec::AVec<f32>)>,
}

/// The `rotary_ndims` of a layer without position (SmolLM3): its first 16 pairs turn by an angle of
/// zero, the identity through the RoPE kernels of a partial rotation (a multiple of 32, as they
/// want).
pub const NOPE_ROTARY_NDIMS: usize = 32;

/// One decoder layer: `x += mixer(norm(x))` then `x += mlp(norm(x))`.
pub struct DecoderLayer {
    pub input_layernorm: crate::core::avec::AVec<BF16>,
    pub block: LayerBlock,
    pub post_attention_layernorm: crate::core::avec::AVec<BF16>,
    pub mlp: Mlp,
    /// Gemma 4 only.
    pub gemma: Option<crate::model::gemma::GemmaLayer>,
    /// `qwen3_5_moe` only, on its sparse layers: the experts. `mlp` is then the shared expert.
    pub qwen_moe: Option<QwenMoe>,
    /// LFM3 only.
    pub lfm3: Option<Lfm3Layer>,
}

/// What an LFM3 layer has beyond LFM2's.
pub struct Lfm3Layer {
    /// `post_operator_norm`, `[hidden]`: the mixer's output normed before its residual (empty
    /// without `use_sandwich_norm`).
    pub post_operator_norm: crate::core::avec::AVec<BF16>,
    /// `post_ffn_norm`, `[hidden]`: the MLP's output normed before its residual (empty without
    /// `use_sandwich_norm`).
    pub post_ffn_norm: crate::core::avec::AVec<BF16>,
    /// `self_attn.gate_proj`, `[num_heads, hidden]`: the gate of each head, `o_h · σ(g_h)` after
    /// the attention; empty on a `conv` layer.
    pub head_gate: PackedBf16R16Weight,
    /// The keys a `sliding_attention` layer sees, itself included; 0 on the full layers.
    pub window: usize,
}

/// The sparse MLP of a `qwen3_5_moe` layer beyond its shared expert (which is the layer's [`Mlp`]):
/// `y = Σ_k w_k · expert_k(x) + σ(shared_gate · x) · shared(x)`.
pub struct QwenMoe {
    /// `mlp.gate`, `[experts, hidden]`, kept exact in bf16 as sixteen-row matrices (a selection
    /// must not move with a quantization): block `b` holds the experts `16b..16b+15`.
    pub router: Vec<PackedBf16R16Weight>,
    /// `mlp.shared_expert_gate`, `[hidden]` bf16: the logit of the shared expert's gate.
    pub shared_gate: crate::core::avec::AVec<BF16>,
    /// Per expert: `[2 · inter, hidden]` (gate rows then up rows) and `[hidden, inter]`.
    pub gate_up: Vec<PackedBf16R16Weight>,
    pub down: Vec<PackedBf16R16Weight>,
    /// `lfm2_moe`: `feed_forward.expert_bias`, `[experts]` f32, added to the sigmoid scores for
    /// the choice only; empty elsewhere.
    pub expert_bias: crate::core::avec::AVec<f32>,
}

/// The mixer of a layer.
pub enum LayerBlock {
    Attention(Attention),
    GatedDeltaNet(GatedDeltaNet),
}

/// Grouped-query attention with per-head Q/K norms, partial RoPE and the
/// fused output gate.
pub struct Attention {
    /// `[2 · num_heads · head_dim, hidden]` with the gate: Q rows first, gate
    /// rows next, once the loader has permuted them.
    pub q_proj: PackedBf16R16Weight,
    /// `[num_kv_heads · head_dim, hidden]`.
    pub k_proj: PackedBf16R16Weight,
    pub v_proj: PackedBf16R16Weight,
    /// `[hidden, num_heads · head_dim]`.
    pub o_proj: PackedBf16R16Weight,
    /// `[head_dim]`, applied to every query head before RoPE.
    pub q_norm: crate::core::avec::AVec<BF16>,
    /// `[head_dim]`, applied to every key head before RoPE.
    pub k_norm: crate::core::avec::AVec<BF16>,
    /// Qwen2.5: the biases of `q_proj`, `k_proj`, `v_proj` (`[q_dim]`, `[kv_dim]`, `[kv_dim]`);
    /// empty elsewhere. With them, `q_norm` and `k_norm` are empty: the RoPE alone follows the
    /// projections.
    pub q_bias: crate::core::avec::AVec<BF16>,
    pub k_bias: crate::core::avec::AVec<BF16>,
    pub v_bias: crate::core::avec::AVec<BF16>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_ndims: usize,
}

/// The GatedDeltaNet mixer of the linear-attention layers.
pub struct GatedDeltaNet {
    /// `[2 · num_k_heads · head_k_dim + num_v_heads · head_v_dim, hidden]`: Q, K, V.
    pub in_proj_qkv: PackedBf16R16Weight,
    /// `[num_v_heads · head_v_dim, hidden]`: the output gate.
    pub in_proj_z: PackedBf16R16Weight,
    /// `[num_v_heads, hidden]` f32: the decay gate `α`.
    pub in_proj_a: crate::core::avec::AVec<f32>,
    /// `[num_v_heads, hidden]` f32: the update gate `β`.
    pub in_proj_b: crate::core::avec::AVec<f32>,
    /// `[conv_dim, conv_kernel]` f32: the depthwise causal conv.
    pub conv1d_weight: crate::core::avec::AVec<f32>,
    /// `[num_v_heads]` f32: `decay = exp(−exp(A_log) · softplus(α + dt_bias))`.
    pub a_log: crate::core::avec::AVec<f32>,
    /// `[num_v_heads]` f32.
    pub dt_bias: crate::core::avec::AVec<f32>,
    /// `[head_v_dim]`: the gated RMSNorm weight.
    pub norm_weight: crate::core::avec::AVec<BF16>,
    /// `[hidden, num_v_heads · head_v_dim]`.
    pub out_proj: PackedBf16R16Weight,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    /// Granite 4's Mamba2 in the DeltaNet's places (`Config::mamba`): `in_proj_qkv` is `[C | B |
    /// x]` (the conv's channels reordered from the checkpoint's `[x | B | C]` so that C, B, x
    /// sit where q, k, v do), `in_proj_z` the gate, `in_proj_a` the step `dt` (`in_proj_b`
    /// zeros, unused), `norm_weight` `[num_v_heads · head_v_dim]` (one norm over the whole
    /// width). Empty and false on Qwen3.5.
    pub mamba: bool,
    /// `[conv_dim]` f32, the conv's bias (Mamba2), in the reordered channel order.
    pub conv1d_bias: crate::core::avec::AVec<f32>,
    /// `[num_v_heads]` f32, the skip `y += D · x` (Mamba2).
    pub d: crate::core::avec::AVec<f32>,
    /// LiquidAI's gated short convolution in the DeltaNet's places
    /// (`Config::is_short_conv_layer`): `in_proj` split at load into `in_proj_z` (the `B`
    /// rows), `in_proj_c` (`C`) and `in_proj_qkv` (`x`), each `[hidden, hidden]`;
    /// `conv1d_weight` `[hidden, conv_kernel_size]`; `out_proj`; no head, gate or norm
    /// (`num_k_heads`, `num_v_heads`, `head_k_dim` 0, `head_v_dim` the hidden width,
    /// `in_proj_a`, `in_proj_b`, `a_log`, `dt_bias`, `norm_weight` empty). False elsewhere,
    /// `in_proj_c` then empty.
    pub short_conv: bool,
    pub in_proj_c: PackedBf16R16Weight,
}

/// The SwiGLU MLP: `down(silu(gate(x)) · up(x))`.
pub struct Mlp {
    /// `[intermediate, hidden]`.
    pub gate_proj: PackedBf16R16Weight,
    pub up_proj: PackedBf16R16Weight,
    /// `[hidden, intermediate]`.
    pub down_proj: PackedBf16R16Weight,
    pub intermediate_size: usize,
}
