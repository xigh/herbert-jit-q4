//! The model's `config.json`, read into the numbers the engine needs.
//!
//! What enters: the JSON of a text checkpoint, the fields possibly nested under `text_config`.
//! The model types served: `qwen3_5` and `qwen3_5_moe` (and their `_text` forms), `gemma4`,
//! `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_vl_moe`, `smollm3`, `granite`, `granitemoe`,
//! `granitemoehybrid`, `ministral3`, `lfm2`, `lfm2_moe`, `lfm3`; each has its reader below.
//! What comes out: one flat [`Config`]. Any other model type is refused by name.
//!
//! What Qwen3.5-4B declares, and what each number drives (the other families fill the same
//! fields from their own keys):
//! - `hidden_size` 2560, `num_hidden_layers` 32, `intermediate_size` 9216;
//! - `num_attention_heads` 16, `num_key_value_heads` 4, `head_dim` 256: the attention layers,
//!   grouped-query (four query heads per KV head);
//! - `layer_types` or `full_attention_interval` 4: which layers are full attention (every fourth)
//!   and which are GatedDeltaNet;
//! - `linear_num_key_heads` 16, `linear_num_value_heads` 32, `linear_key_head_dim` 128,
//!   `linear_value_head_dim` 128, `linear_conv_kernel_dim` 4: the GatedDeltaNet mixer;
//! - `rope_parameters.partial_rotary_factor` 0.25 on `head_dim` 256: RoPE rotates 64 dimensions of
//!   each head, `rope_theta` 1e7;
//! - `attn_output_gate`: `q_proj` carries `[Q; gate]` and the attention output is multiplied by
//!   `sigmoid(gate)`;
//! - `rms_norm_eps`, and the norms are zero-centred: `(1 + w) · rms(x)`.
//!
//! `qwen3_5_moe` (Qwen3.5/3.6-35B-A3B) replaces the MLP of its sparse layers by a mixture of
//! experts ([`QwenMoeConfig`]): `num_experts` SwiGLU experts of `moe_intermediate_size`,
//! `num_experts_per_tok` per token, and a shared SwiGLU expert of `shared_expert_intermediate_size`
//! gated by `sigmoid(shared_expert_gate · x)`. The shared expert is run as the layer's dense MLP:
//! `intermediate_size` is then its width (the widest over the layers when `mlp_only_layers` keeps
//! dense ones).

use crate::core::error::{HerbertError, Result};
use serde_json::Value;
use std::path::Path;

/// How a norm weight is applied. Qwen3.5 stores every norm weight zero-centred
/// (`GemmaRMSNorm`): the kernel computes `(1 + w) · rms(x)`, not `w · rms(x)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RMSNorm,
    GemmaRMSNorm,
}

/// What a layer's mixer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
    /// LiquidAI's gated short convolution (`lfm2`, `lfm2_moe`, the `conv` layers): `[B | C | x] =
    /// in_proj(xn)`, `y = C · conv(B · x)` (a causal depthwise conv of
    /// `linear_conv_kernel_dim` taps, no bias), `out_proj(y)`.
    ShortConv,
}

/// What a Gemma 4 text checkpoint (`gemma4`, `gemma4_text`, `gemma4_unified`; E2B to 31B) adds to
/// [`Config`]: two kinds of attention layer (a sliding window and a full one, with their own head
/// width and RoPE), the layers that read the KV cache of an earlier layer, the per-layer input
/// branch, the logit soft cap.
#[derive(Debug, Clone)]
pub struct GemmaConfig {
    /// Per layer: sliding-window attention (true) or full attention (false).
    pub layer_sliding: Vec<bool>,
    /// Positions a sliding layer sees (the layer itself included).
    pub sliding_window: usize,
    /// `head_dim` of the full-attention layers (`Config::head_dim` is the sliding one).
    pub global_head_dim: usize,
    /// Key/value heads of the full-attention layers.
    pub global_kv_heads: usize,
    /// The last `num_kv_shared_layers` layers own no cache: they read the one of the last earlier
    /// layer of their kind.
    pub num_kv_shared_layers: usize,
    /// Shared layers have an MLP twice as wide.
    pub use_double_wide_mlp: bool,
    /// Width of the per-layer input (`hidden_size_per_layer_input`), 0 = no such branch.
    pub pl_dim: usize,
    pub vocab_size_per_layer_input: usize,
    /// `final_logit_softcapping`.
    pub final_logit_softcap: f32,
    /// RoPE base of the sliding layers and of the full ones.
    pub rope_theta_sliding: f32,
    pub rope_theta_full: f32,
    /// Fraction of the dimensions of a full layer's head that the (proportional) RoPE rotates.
    pub full_partial_rotary_factor: f32,
    /// `attention_k_eq_v`: the full-attention layers have no `v_proj`, their values are their
    /// (unnormed) keys.
    pub k_eq_v: bool,
    /// The mixture-of-experts block that runs in parallel to the dense MLP of every layer (the
    /// 26B-A4B).
    pub moe: Option<MoeConfig>,
}

/// The mixture-of-experts block of a Gemma 4 layer.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub num_experts: usize,
    /// Experts a token uses.
    pub top_k: usize,
    /// Width of an expert (`moe_intermediate_size`).
    pub inter: usize,
}

/// The sparse MLP of a `qwen3_5_moe` layer (`Qwen3_5MoeSparseMoeBlock`).
#[derive(Debug, Clone)]
pub struct QwenMoeConfig {
    pub num_experts: usize,
    /// Experts a token uses (`num_experts_per_tok`).
    pub top_k: usize,
    /// Width of an expert (`moe_intermediate_size`).
    pub inter: usize,
    /// Width of the shared expert (`shared_expert_intermediate_size`).
    pub shared_inter: usize,
    /// `norm_topk_prob`: the `top_k` probabilities divided by their sum (true by default, as the
    /// module does).
    pub norm_topk_prob: bool,
    /// Per layer: an expert block (true) or the dense MLP of `intermediate_size`
    /// (`mlp_only_layers`, `decoder_sparse_step`).
    pub sparse: Vec<bool>,
    /// `intermediate_size` of the dense layers (0 when every layer is sparse).
    pub dense_inter: usize,
    /// LiquidAI's route (`lfm2_moe`, `Lfm2MoeTopKRouter`) in place of the softmax: `s = σ(logit)`
    /// per expert, the `top_k` best of `s + expert_bias`, their `s` divided by `Σs + 1e-6`
    /// (`norm_topk_prob`), times this `routed_scaling_factor`. No shared expert. `None`
    /// elsewhere.
    pub lfm_scale: Option<f32>,
}

/// What LiquidAI's LFM3 (`lfm3`) adds to LFM2's reading (`Config::from_lfm2`): attention layers
/// of two kinds, each head's output gated, the outputs of the mixer and of the MLP normed before
/// the residual.
#[derive(Debug, Clone)]
pub struct Lfm3Config {
    /// Per layer: `sliding_attention` (true), else `full_attention` or `conv`.
    pub sliding: Vec<bool>,
    /// The keys a sliding layer sees, itself included (`sliding_window`).
    pub sliding_window: usize,
    /// The RoPE base of the sliding layers, which rotate the whole head
    /// (`rope_parameters.sliding_attention`).
    pub rope_theta_sliding: f32,
    /// The pairs the full layers rotate, `partial_rotary_factor · head_dim / 2`, at the
    /// frequencies of the whole head (`θ^(−2i/head_dim)`, `Config::rope_theta`); the pairs
    /// after them turn by zero (the zero-tail partial RoPE).
    pub rope_angles: usize,
    /// `use_sandwich_norm`: `post_operator_norm` on the mixer's output and `post_ffn_norm` on the
    /// MLP's, before each residual.
    pub sandwich_norm: bool,
}

/// The four multipliers of a Granite 3 checkpoint (`granitemoe`). The loader folds three of them
/// into the weights (`embedding_multiplier` into the embedding table, `residual_multiplier` into
/// `o_proj` and the experts' `down`, `1 / logits_scaling` into the language head, each product
/// rounded to bf16 once); `attention_multiplier` is the attention scale.
#[derive(Debug, Clone, Copy)]
pub struct GraniteConfig {
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
    pub attention_multiplier: f32,
}

/// YaRN's RoPE (`rope_type` `yarn`, Ministral 3): the frequencies past the pretraining context are
/// interpolated by `factor`, the fast ones kept, a linear ramp between the two over the dimensions
/// whose number of rotations in `original_max_position_embeddings` lies between `beta_slow` and
/// `beta_fast`; cos and sin are multiplied by `attention_factor`. Computed as transformers'
/// `_compute_yarn_parameters` does.
#[derive(Debug, Clone, Copy)]
pub struct Yarn {
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub attention_factor: f64,
    pub truncate: bool,
}

impl Yarn {
    /// The inverse frequencies of the `rotary_ndims / 2` pairs.
    pub fn inv_freq(&self, theta: f32, rotary_ndims: usize) -> Vec<f32> {
        let dim = rotary_ndims as f64;
        let correction = |rotations: f64| {
            dim * (self.original_max_position_embeddings as f64
                / (rotations * 2.0 * std::f64::consts::PI))
                .ln()
                / (2.0 * (theta as f64).ln())
        };
        let (mut low, mut high) = (correction(self.beta_fast), correction(self.beta_slow));
        if self.truncate {
            (low, high) = (low.floor(), high.ceil());
        }
        let (low, mut high) = (low.max(0.0), high.min(dim - 1.0));
        if low == high {
            high += 0.001;
        }
        (0..rotary_ndims / 2)
            .map(|i| {
                let pos_freq = theta.powf((2 * i) as f32 / rotary_ndims as f32);
                let extrapolation = 1.0 / pos_freq;
                let interpolation = 1.0 / (self.factor as f32 * pos_freq);
                let ramp = (((i as f32) - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
                let keep = 1.0 - ramp;
                interpolation * (1.0 - keep) + extrapolation * keep
            })
            .collect()
    }
}

impl GemmaConfig {
    /// Index of the first layer that reads another layer's cache (`num_layers` when none).
    pub fn first_shared(&self) -> usize {
        self.layer_sliding.len() - self.num_kv_shared_layers
    }

    /// The layer whose cache layer `l` reads: itself when it owns one, otherwise the last earlier
    /// layer of its kind.
    pub fn kv_source(&self, l: usize) -> usize {
        let first = self.first_shared();
        if self.num_kv_shared_layers == 0 || l < first {
            return l;
        }
        (0..first)
            .rev()
            .find(|&i| self.layer_sliding[i] == self.layer_sliding[l])
            .unwrap_or(l)
    }

    pub fn moe_config(&self) -> Option<MoeConfig> {
        self.moe
    }

    pub fn head_dim_of(&self, l: usize, sliding_head_dim: usize) -> usize {
        if self.layer_sliding[l] {
            sliding_head_dim
        } else {
            self.global_head_dim
        }
    }

    pub fn intermediate_of(&self, l: usize, base: usize) -> usize {
        if self.use_double_wide_mlp && self.num_kv_shared_layers > 0 && l >= self.first_shared() {
            2 * base
        } else {
            base
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub norm_type: NormType,
    pub rope_theta: f32,
    /// Dimensions of each head rotated by RoPE: `head_dim · partial_rotary_factor`.
    pub rotary_ndims: usize,
    /// Positions the RoPE tables cover: the model's context ceiling.
    pub max_position_embeddings: usize,
    /// The language head is the embedding table.
    pub tie_word_embeddings: bool,
    /// The activation of the MLP and of the DeltaNet conv; `silu` is what the
    /// kernels emit.
    pub hidden_act: String,
    /// `q_proj` is `[Q; gate]` and the attention output is gated by `sigmoid(gate)`.
    pub attn_output_gate: bool,
    /// Each query and key head is RMS-normed before the RoPE (`q_norm`, `k_norm`): Qwen3.5 and
    /// Gemma 4, not Qwen2.5.
    pub qk_norm: bool,
    /// `q_proj`, `k_proj` and `v_proj` have a bias (Qwen2.5).
    pub qkv_bias: bool,
    /// The mixer of each layer, in order.
    pub layer_types: Vec<LayerType>,
    // ── the GatedDeltaNet mixer ──
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    /// Gemma 4 only.
    pub gemma: Option<GemmaConfig>,
    /// `qwen3_5_moe` only: the experts of the sparse layers. Granite 3's mixture too, without a
    /// shared expert (`shared_inter` 0: the layer has no dense MLP).
    pub qwen_moe: Option<QwenMoeConfig>,
    /// Granite 3 and 4 only.
    pub granite: Option<GraniteConfig>,
    /// The RoPE is YaRN's (Ministral 3).
    pub yarn: Option<Yarn>,
    /// The linear layers are Mamba2 (Granite 4, `granitemoehybrid`), not GatedDeltaNet: the
    /// `linear_*` fields give its shape as the DeltaNet's (one key head per group of
    /// `mamba_n_groups`, the state `d_state` wide; the value heads are Mamba's heads, `d_head`
    /// wide).
    pub mamba: bool,
    /// No position in the attention (`position_embedding_type` `nope`, Granite 4 H): the RoPE
    /// tables are the identity.
    pub nope: bool,
    /// Per layer, no position in that layer's attention (SmolLM3, `no_rope_layers` 0); empty when
    /// every layer has it.
    pub no_rope: Vec<bool>,
    /// LFM3 only: the sliding layers, the zero-tail RoPE of the full ones, the gate per head
    /// (`self_attn.gate_proj`: `o_h · σ(g_h)`), the sandwich norms.
    pub lfm3: Option<Lfm3Config>,
    /// The text model of Qwen3-VL (`qwen3_vl_moe_text`): the multimodal RoPE and the deepstack
    /// inputs. `None` elsewhere.
    pub qwen_vl: Option<QwenVlConfig>,
}

/// The text model of a Qwen3-VL checkpoint (`qwen3_vl_moe`, `Qwen3VLMoeTextModel`): what differs
/// from a text Qwen3.
///
/// - The multimodal RoPE (`rope_scaling.mrope_section`, `mrope_interleaved`): each of the `head_dim
///   / 2` frequencies of a head turns by the position of one axis, time `t`, height `h` or width
///   `w` ([`Self::mrope_axis`]). A text token has `t = h = w`, and its rotation is then the plain
///   RoPE of `rope_theta`; an image token has its own `(t, h, w)`
///   ([`crate::model::loader::mrope_row`]).
/// - The deepstack: after layer `l < deepstack_layers`, the rows of the image tokens receive a
///   vector of the vision tower (`hidden[visual_pos_masks] += deepstack_visual_embeds[l]`, in bf16,
///   `_deepstack_process`).
#[derive(Debug, Clone, PartialEq)]
pub struct QwenVlConfig {
    /// Frequencies of a half head turned by `t`, `h`, `w` (`mrope_section`, `[24, 20, 20]`); their
    /// sum is `head_dim / 2`.
    pub mrope_section: [usize; 3],
    /// `mrope_interleaved` (Qwen3-VL): `h` turns the frequencies `1, 4, …, 3·section[1] − 2`, `w`
    /// the frequencies `2, 5, …, 3·section[2] − 1`, `t` the others. Otherwise (Qwen2-VL's
    /// layout) three consecutive spans.
    pub interleaved: bool,
    /// Layers after which the deepstack vectors are added
    /// (`len(vision_config.deepstack_visual_indexes)`, 3).
    pub deepstack_layers: usize,
}

impl QwenVlConfig {
    /// The axis (0 `t`, 1 `h`, 2 `w`) whose position turns frequency `i` of a half head
    /// (`Qwen3VLMoeTextRotaryEmbedding.recomposition_frequencies`).
    pub fn mrope_axis(&self, i: usize) -> usize {
        let s = &self.mrope_section;
        if self.interleaved {
            if i % 3 == 1 && i < 3 * s[1] {
                1
            } else if i % 3 == 2 && i < 3 * s[2] {
                2
            } else {
                0
            }
        } else if i < s[0] {
            0
        } else if i < s[0] + s[1] {
            1
        } else {
            2
        }
    }
}

impl Config {
    /// Reads `config.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Parses the JSON text. The text fields may sit at the top level or under
    /// `text_config` (the multimodal packaging of Qwen3.5); the top level wins
    /// for `tie_word_embeddings` when both are present, as Hugging Face does.
    pub fn from_json(content: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(content)
            .map_err(|e| HerbertError::Config(format!("config.json: {e}")))?;
        let text = root.get("text_config").unwrap_or(&root);
        let field = |key: &str| text.get(key).or_else(|| root.get(key));
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            field(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        let model_type = field("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| missing("model_type"))?
            .to_string();
        if matches!(
            model_type.as_str(),
            "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text"
        ) {
            return Self::from_gemma(&root, text, model_type);
        }
        if matches!(
            model_type.as_str(),
            "qwen2" | "qwen3" | "smollm3" | "qwen3_moe" | "qwen3_vl_moe" | "qwen3_vl_moe_text"
        ) {
            return Self::from_qwen2(&root, text, model_type);
        }
        if model_type == "ministral3" {
            return Self::from_ministral3(&root, text, model_type);
        }
        if model_type == "granitemoehybrid" {
            return Self::from_granite_hybrid(&root, model_type);
        }
        if model_type == "granitemoe" || model_type == "granite" {
            return Self::from_granite(&root, model_type);
        }
        if model_type == "lfm2" || model_type == "lfm2_moe" || model_type == "lfm3" {
            // LFM3-VL (`lfm3_vl`): the text model under `text_config`.
            return Self::from_lfm2(text, model_type);
        }
        if !["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]
            .contains(&model_type.as_str())
        {
            return Err(HerbertError::Config(format!(
                "model_type {model_type:?}: this engine serves qwen3_5 and qwen3_5_moe (text), gemma4, qwen2, qwen3, qwen3_moe, qwen3_vl_moe (text), smollm3, granite, granitemoe, granitemoehybrid, ministral3, lfm2, lfm2_moe, lfm3"
            )));
        }
        let head_dim = usize_of("head_dim")?;
        let num_layers = usize_of("num_hidden_layers")?;
        // `qwen3_5_moe`: the sparse layers (every one unless `mlp_only_layers` or
        // `decoder_sparse_step` say otherwise); their dense MLP is the shared expert.
        let qwen_moe = if model_type.starts_with("qwen3_5_moe") {
            let only: Vec<usize> = field("mlp_only_layers")
                .and_then(Value::as_array)
                .map(|v| {
                    v.iter()
                        .filter_map(Value::as_u64)
                        .map(|n| n as usize)
                        .collect()
                })
                .unwrap_or_default();
            let step = usize_of("decoder_sparse_step").unwrap_or(1).max(1);
            let sparse: Vec<bool> = (0..num_layers)
                .map(|l| !only.contains(&l) && (l + 1).is_multiple_of(step))
                .collect();
            let dense_inter = if sparse.iter().all(|&s| s) {
                0
            } else {
                usize_of("intermediate_size")?
            };
            Some(QwenMoeConfig {
                num_experts: usize_of("num_experts")?,
                top_k: usize_of("num_experts_per_tok")?,
                inter: usize_of("moe_intermediate_size")?,
                shared_inter: usize_of("shared_expert_intermediate_size")?,
                norm_topk_prob: field("norm_topk_prob")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                sparse,
                dense_inter,
                lfm_scale: None,
            })
        } else {
            None
        };
        // RoPE: `rope_parameters` on Qwen3.5, the flat keys on older packagings.
        let rope = field("rope_parameters");
        let rope_field = |key: &str| {
            rope.and_then(|r| r.get(key))
                .or_else(|| field(key))
                .and_then(Value::as_f64)
        };
        let rope_theta = rope_field("rope_theta").ok_or_else(|| missing("rope_theta"))? as f32;
        let partial_rotary_factor = rope_field("partial_rotary_factor").unwrap_or(1.0);
        let rotary_ndims = (head_dim as f64 * partial_rotary_factor).round() as usize;
        // Which layer is which mixer: the explicit list, or every `interval`-th layer is attention.
        let layer_types: Vec<LayerType> = match field("layer_types").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|v| match v.as_str() {
                    Some("full_attention") => Ok(LayerType::FullAttention),
                    Some("linear_attention") => Ok(LayerType::LinearAttention),
                    other => Err(HerbertError::Config(format!(
                        "config.json: layer_types holds {other:?}"
                    ))),
                })
                .collect::<Result<_>>()?,
            None => {
                let interval = usize_of("full_attention_interval").unwrap_or(1).max(1);
                (0..num_layers)
                    .map(|l| {
                        if (l + 1) % interval == 0 {
                            LayerType::FullAttention
                        } else {
                            LayerType::LinearAttention
                        }
                    })
                    .collect()
            }
        };
        let has_linear = layer_types.contains(&LayerType::LinearAttention);
        let linear = |key: &str| -> Result<usize> {
            if has_linear {
                usize_of(key)
            } else {
                Ok(0)
            }
        };
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size: usize_of("hidden_size")?,
            num_layers,
            num_attention_heads: usize_of("num_attention_heads")?,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: match &qwen_moe {
                Some(m) => m.shared_inter.max(m.dense_inter),
                None => usize_of("intermediate_size")?,
            },
            rms_norm_eps: field("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            norm_type: NormType::GemmaRMSNorm,
            rope_theta,
            rotary_ndims,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .or_else(|| text.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hidden_act: field("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            attn_output_gate: field("attn_output_gate")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            qk_norm: true,
            qkv_bias: false,
            layer_types,
            linear_num_key_heads: linear("linear_num_key_heads")?,
            linear_num_value_heads: linear("linear_num_value_heads")?,
            linear_key_head_dim: linear("linear_key_head_dim")?,
            linear_value_head_dim: linear("linear_value_head_dim")?,
            linear_conv_kernel_dim: linear("linear_conv_kernel_dim")?,
            gemma: None,
            qwen_moe,
            granite: None,
            yarn: None,
            mamba: false,
            nope: false,
            no_rope: Vec::new(),
            lfm3: None,
            qwen_vl: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// The text model of a Gemma 4 checkpoint (`text_config` of `gemma4`, or a `gemma4_text` file).
    fn from_gemma(root: &Value, text: &Value, model_type: String) -> Result<Self> {
        let field = |key: &str| text.get(key).or_else(|| root.get(key));
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            field(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        let refuse = |what: &str| Err(HerbertError::Config(format!("config.json: {what}")));
        let moe = if field("enable_moe_block")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            Some(MoeConfig {
                num_experts: usize_of("num_experts")?,
                top_k: usize_of("top_k_experts")?,
                inter: usize_of("moe_intermediate_size")?,
            })
        } else {
            None
        };
        let k_eq_v = field("attention_k_eq_v")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let num_layers = usize_of("num_hidden_layers")?;
        let layer_sliding: Vec<bool> = field("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| missing("layer_types"))?
            .iter()
            .map(|v| match v.as_str() {
                Some("sliding_attention") => Ok(true),
                Some("full_attention") => Ok(false),
                other => Err(HerbertError::Config(format!(
                    "config.json: layer_types holds {other:?}"
                ))),
            })
            .collect::<Result<_>>()?;
        let rope = field("rope_parameters").ok_or_else(|| missing("rope_parameters"))?;
        let rope_of = |kind: &str, key: &str| {
            rope.get(kind)
                .and_then(|r| r.get(key))
                .and_then(Value::as_f64)
        };
        let head_dim = usize_of("head_dim")?;
        let kv_heads = usize_of("num_key_value_heads")?;
        let gemma = GemmaConfig {
            layer_sliding,
            sliding_window: usize_of("sliding_window")?,
            global_head_dim: usize_of("global_head_dim")?,
            global_kv_heads: field("num_global_key_value_heads")
                .and_then(Value::as_u64)
                .map_or(kv_heads, |n| n as usize),
            num_kv_shared_layers: usize_of("num_kv_shared_layers").unwrap_or(0),
            use_double_wide_mlp: field("use_double_wide_mlp")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            pl_dim: usize_of("hidden_size_per_layer_input").unwrap_or(0),
            vocab_size_per_layer_input: usize_of("vocab_size_per_layer_input").unwrap_or(0),
            final_logit_softcap: field("final_logit_softcapping")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32,
            rope_theta_sliding: rope_of("sliding_attention", "rope_theta")
                .ok_or_else(|| missing("rope_parameters.sliding_attention.rope_theta"))?
                as f32,
            rope_theta_full: rope_of("full_attention", "rope_theta")
                .ok_or_else(|| missing("rope_parameters.full_attention.rope_theta"))?
                as f32,
            full_partial_rotary_factor: rope_of("full_attention", "partial_rotary_factor")
                .unwrap_or(1.0) as f32,
            k_eq_v,
            moe,
        };
        if gemma.layer_sliding.len() != num_layers {
            return refuse("layer_types does not have one entry per layer");
        }
        if gemma.num_kv_shared_layers >= num_layers {
            return refuse("num_kv_shared_layers leaves no layer with a cache");
        }
        let hidden_act = field("hidden_activation")
            .and_then(Value::as_str)
            .unwrap_or("gelu_pytorch_tanh")
            .to_string();
        if hidden_act != "gelu_pytorch_tanh" {
            return refuse(&format!(
                "hidden_activation {hidden_act:?} (gelu_pytorch_tanh expected)"
            ));
        }
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size: usize_of("hidden_size")?,
            num_layers,
            num_attention_heads: usize_of("num_attention_heads")?,
            num_key_value_heads: kv_heads,
            head_dim,
            intermediate_size: usize_of("intermediate_size")?,
            rms_norm_eps: field("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            // Gemma 4 stores its norm weights as they are applied: `w · rms(x)`.
            norm_type: NormType::RMSNorm,
            rope_theta: gemma.rope_theta_sliding,
            rotary_ndims: head_dim,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .or_else(|| text.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
            hidden_act,
            attn_output_gate: false,
            qk_norm: true,
            qkv_bias: false,
            layer_types: vec![LayerType::FullAttention; num_layers],
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            gemma: Some(gemma),
            qwen_moe: None,
            granite: None,
            yarn: None,
            mamba: false,
            nope: false,
            no_rope: Vec::new(),
            lfm3: None,
            qwen_vl: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// A Qwen2.5 checkpoint (`qwen2`, `Qwen2ForCausalLM`): every layer is attention, grouped-query,
    /// with biases on `q_proj`, `k_proj`, `v_proj`, no per-head norm, no gate, the whole head
    /// rotated, the norms `w · rms(x)`. The sliding window is refused when `use_sliding_window`
    /// turns it on (it is off on every Qwen2.5 checkpoint).
    ///
    /// The same reader serves two neighbours. `qwen3` (Qwen3, Chocolatine 2): no bias, `q_norm` and
    /// `k_norm` on every head (`w · rms(x)`), `head_dim` declared. `smollm3` (SmolLM3): no
    /// bias, no per-head norm, and no position in the layers that `no_rope_layers` marks 0
    /// (every fourth).
    ///
    /// `qwen3_moe` (Qwen3-30B-A3B, `Qwen3MoeForCausalLM`) is Qwen3 with a mixture of experts on its
    /// sparse layers (`mlp_only_layers`, `decoder_sparse_step`): `num_experts` SwiGLU experts
    /// of `moe_intermediate_size`, `num_experts_per_tok` per token, the softmax over all
    /// experts then its `top_k` (renormalized when `norm_topk_prob`, false by default as in
    /// `Qwen3MoeConfig`), no shared expert. `qwen3_vl_moe` (Qwen3-VL-30B-A3B,
    /// its `text_config`, `qwen3_vl_moe_text`) is the same text model, its route always
    /// renormalized (`Qwen3VLMoeTextTopKRouter`), with the multimodal RoPE and the deepstack
    /// inputs ([`QwenVlConfig`]).
    fn from_qwen2(full: &Value, root: &Value, model_type: String) -> Result<Self> {
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            root.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        if root.get("use_sliding_window").and_then(Value::as_bool) == Some(true) {
            return Err(HerbertError::Config(format!(
                "config.json: {model_type} with use_sliding_window is not served"
            )));
        }
        if model_type != "qwen2"
            && (root.get("attention_bias").and_then(Value::as_bool) == Some(true)
                || root.get("mlp_bias").and_then(Value::as_bool) == Some(true))
        {
            return Err(HerbertError::Config(format!(
                "config.json: {model_type} with attention_bias or mlp_bias is not served"
            )));
        }
        let hidden_size = usize_of("hidden_size")?;
        let num_attention_heads = usize_of("num_attention_heads")?;
        let head_dim = usize_of("head_dim").unwrap_or(hidden_size / num_attention_heads.max(1));
        let num_layers = usize_of("num_hidden_layers")?;
        // SmolLM3: `no_rope_layers[l]` is 1 where the layer rotates, 0 where it has no position.
        let no_rope: Vec<bool> = root
            .get("no_rope_layers")
            .and_then(Value::as_array)
            .map(|l| l.iter().map(|v| v.as_u64() == Some(0)).collect())
            .unwrap_or_default();
        if !no_rope.is_empty() && no_rope.len() != num_layers {
            return Err(HerbertError::Config(
                "config.json: no_rope_layers does not have one entry per layer".into(),
            ));
        }
        let vl = model_type.starts_with("qwen3_vl_moe");
        let (qk_norm, qkv_bias) = (
            model_type == "qwen3" || model_type == "qwen3_moe" || vl,
            model_type == "qwen2",
        );
        let qwen_moe = if model_type == "qwen3_moe" || vl {
            let only: Vec<usize> = root
                .get("mlp_only_layers")
                .and_then(Value::as_array)
                .map(|v| {
                    v.iter()
                        .filter_map(Value::as_u64)
                        .map(|n| n as usize)
                        .collect()
                })
                .unwrap_or_default();
            let step = usize_of("decoder_sparse_step").unwrap_or(1).max(1);
            let sparse: Vec<bool> = (0..num_layers)
                .map(|l| !only.contains(&l) && (l + 1).is_multiple_of(step))
                .collect();
            let dense_inter = if sparse.iter().all(|&s| s) {
                0
            } else {
                usize_of("intermediate_size")?
            };
            Some(QwenMoeConfig {
                num_experts: usize_of("num_experts")?,
                top_k: usize_of("num_experts_per_tok")?,
                inter: usize_of("moe_intermediate_size")?,
                shared_inter: 0,
                norm_topk_prob: vl
                    || root
                        .get("norm_topk_prob")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                sparse,
                dense_inter,
                lfm_scale: None,
            })
        } else {
            None
        };
        // The multimodal RoPE: `rope_scaling` on the checkpoints, `rope_parameters` in transformers
        // 5.
        let rope = root
            .get("rope_parameters")
            .or_else(|| root.get("rope_scaling"));
        let qwen_vl = if vl {
            let section: Vec<usize> = rope
                .and_then(|r| r.get("mrope_section"))
                .and_then(Value::as_array)
                .map(|v| {
                    v.iter()
                        .filter_map(Value::as_u64)
                        .map(|n| n as usize)
                        .collect()
                })
                .unwrap_or_else(|| vec![24, 20, 20]);
            let [t, h, w] = section[..] else {
                return Err(HerbertError::Config(
                    "config.json: mrope_section does not hold three values".into(),
                ));
            };
            let deepstack_layers = full
                .get("vision_config")
                .and_then(|v| v.get("deepstack_visual_indexes"))
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            Some(QwenVlConfig {
                mrope_section: [t, h, w],
                interleaved: rope
                    .and_then(|r| r.get("mrope_interleaved"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                deepstack_layers,
            })
        } else {
            None
        };
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: if qwen_moe.as_ref().is_some_and(|m| m.dense_inter == 0) {
                0
            } else {
                usize_of("intermediate_size")?
            },
            rms_norm_eps: root
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            norm_type: NormType::RMSNorm,
            rope_theta: root
                .get("rope_theta")
                .or_else(|| rope.and_then(|r| r.get("rope_theta")))
                .and_then(Value::as_f64)
                .ok_or_else(|| missing("rope_theta"))? as f32,
            rotary_ndims: head_dim,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hidden_act: root
                .get("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            attn_output_gate: false,
            qk_norm,
            qkv_bias,
            layer_types: vec![LayerType::FullAttention; num_layers],
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            gemma: None,
            qwen_moe,
            granite: None,
            yarn: None,
            mamba: false,
            nope: false,
            no_rope,
            lfm3: None,
            qwen_vl,
        };
        config.validate()?;
        Ok(config)
    }

    /// A Granite checkpoint, dense (`granite`, `GraniteForCausalLM`: Granite 3.x 2B and 8B, 4.1,
    /// 4.2) or a mixture of experts (`granitemoe`, `GraniteMoeForCausalLM`). Every layer is
    /// attention (grouped-query, no bias, no per-head norm, the whole head rotated), then a
    /// SwiGLU MLP of `intermediate_size`, or `num_local_experts` SwiGLU experts
    /// of `intermediate_size`, `num_experts_per_tok` per token, the softmax over the chosen logits
    /// (equal to the renormalized `top_k` of the full softmax, the route of `qwen3_5_moe`), no
    /// shared expert and no dense MLP.
    fn from_granite(root: &Value, model_type: String) -> Result<Self> {
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            root.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        let f32_of = |key: &str, default: f64| {
            root.get(key).and_then(Value::as_f64).unwrap_or(default) as f32
        };
        if root.get("attention_bias").and_then(Value::as_bool) == Some(true)
            || root.get("mlp_bias").and_then(Value::as_bool) == Some(true)
        {
            return Err(HerbertError::Config(format!(
                "config.json: {model_type} with attention_bias or mlp_bias is not served"
            )));
        }
        let moe = model_type == "granitemoe";
        if root
            .get("shared_intermediate_size")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
        {
            return Err(HerbertError::Config(format!(
                "config.json: {model_type} with a shared expert is not served"
            )));
        }
        let hidden_size = usize_of("hidden_size")?;
        let num_attention_heads = usize_of("num_attention_heads")?;
        let head_dim = hidden_size / num_attention_heads.max(1);
        let num_layers = usize_of("num_hidden_layers")?;
        let inter = usize_of("intermediate_size")?;
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: if moe { 0 } else { inter },
            rms_norm_eps: f32_of("rms_norm_eps", 1e-6),
            norm_type: NormType::RMSNorm,
            rope_theta: root
                .get("rope_theta")
                .or_else(|| {
                    root.get("rope_parameters")
                        .and_then(|r| r.get("rope_theta"))
                })
                .and_then(Value::as_f64)
                .ok_or_else(|| missing("rope_theta"))? as f32,
            rotary_ndims: head_dim,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hidden_act: root
                .get("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            attn_output_gate: false,
            qk_norm: false,
            qkv_bias: false,
            layer_types: vec![LayerType::FullAttention; num_layers],
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            gemma: None,
            qwen_moe: if moe {
                Some(QwenMoeConfig {
                    num_experts: usize_of("num_local_experts")?,
                    top_k: usize_of("num_experts_per_tok")?,
                    inter,
                    shared_inter: 0,
                    norm_topk_prob: true,
                    sparse: vec![true; num_layers],
                    dense_inter: 0,
                    lfm_scale: None,
                })
            } else {
                None
            },
            granite: Some(GraniteConfig {
                embedding_multiplier: f32_of("embedding_multiplier", 1.0),
                residual_multiplier: f32_of("residual_multiplier", 1.0),
                logits_scaling: f32_of("logits_scaling", 1.0),
                attention_multiplier: f32_of(
                    "attention_multiplier",
                    1.0 / (head_dim as f64).sqrt(),
                ),
            }),
            yarn: None,
            mamba: false,
            nope: false,
            no_rope: Vec::new(),
            lfm3: None,
            qwen_vl: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// The text model of a Ministral 3 checkpoint (`text_config` of `mistral3`, `model_type`
    /// `ministral3`; the vision tower is not read): every layer is attention, grouped-query, no
    /// bias, no per-head norm, the whole head rotated by YaRN's RoPE, the norms `w · rms(x)`, a
    /// SwiGLU MLP. Past `original_max_position_embeddings` the queries are multiplied by `1 + β
    /// · ln(1 + ⌊pos / original⌋)` (`llama_4_scaling_beta`); below it that factor is exactly 1, so
    /// the positions are bounded there while the engine does not emit it.
    fn from_ministral3(root: &Value, text: &Value, model_type: String) -> Result<Self> {
        let field = |key: &str| text.get(key).or_else(|| root.get(key));
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            field(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        if field("sliding_window").and_then(Value::as_u64).is_some() {
            return Err(HerbertError::Config(
                "config.json: ministral3 with a sliding window is not served".into(),
            ));
        }
        let rope = field("rope_parameters")
            .or_else(|| field("rope_scaling"))
            .ok_or_else(|| missing("rope_parameters"))?;
        let rope_f64 = |key: &str| rope.get(key).and_then(Value::as_f64);
        let rope_theta = rope_f64("rope_theta")
            .or_else(|| field("rope_theta").and_then(Value::as_f64))
            .ok_or_else(|| missing("rope_theta"))?;
        let mut max_position_embeddings = usize_of("max_position_embeddings")?;
        let yarn = match rope.get("rope_type").and_then(Value::as_str) {
            Some("yarn") => {
                let factor = rope_f64("factor").ok_or_else(|| missing("rope_parameters.factor"))?;
                let original = rope
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| missing("rope_parameters.original_max_position_embeddings"))?
                    as usize;
                let mscale = |scale: f64, m: f64| {
                    if scale <= 1.0 {
                        1.0
                    } else {
                        0.1 * m * scale.ln() + 1.0
                    }
                };
                let attention_factor = match (
                    rope_f64("attention_factor"),
                    rope_f64("mscale"),
                    rope_f64("mscale_all_dim"),
                ) {
                    (Some(a), _, _) => a,
                    (None, Some(m), Some(all)) if m != 0.0 && all != 0.0 => {
                        mscale(factor, m) / mscale(factor, all)
                    }
                    _ => mscale(factor, 1.0),
                };
                if rope_f64("llama_4_scaling_beta").unwrap_or(0.0) != 0.0 {
                    max_position_embeddings = max_position_embeddings.min(original);
                }
                Some(Yarn {
                    factor,
                    original_max_position_embeddings: original,
                    beta_fast: rope_f64("beta_fast").filter(|&b| b != 0.0).unwrap_or(32.0),
                    beta_slow: rope_f64("beta_slow").filter(|&b| b != 0.0).unwrap_or(1.0),
                    attention_factor,
                    truncate: rope
                        .get("truncate")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                })
            }
            None | Some("default") => None,
            Some(other) => {
                return Err(HerbertError::Config(format!(
                    "config.json: ministral3 with rope_type {other:?} is not served"
                )))
            }
        };
        let hidden_size = usize_of("hidden_size")?;
        let num_attention_heads = usize_of("num_attention_heads")?;
        let head_dim = usize_of("head_dim").unwrap_or(hidden_size / num_attention_heads.max(1));
        let num_layers = usize_of("num_hidden_layers")?;
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: usize_of("intermediate_size")?,
            rms_norm_eps: field("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            norm_type: NormType::RMSNorm,
            rope_theta: rope_theta as f32,
            rotary_ndims: head_dim,
            max_position_embeddings,
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hidden_act: field("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            attn_output_gate: false,
            qk_norm: false,
            qkv_bias: false,
            layer_types: vec![LayerType::FullAttention; num_layers],
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            gemma: None,
            qwen_moe: None,
            granite: None,
            yarn,
            mamba: false,
            nope: false,
            no_rope: Vec::new(),
            lfm3: None,
            qwen_vl: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// A Granite 4 checkpoint (`granitemoehybrid`, `GraniteMoeHybridForCausalLM`): each layer is
    /// attention (grouped-query, no bias, RoPE or none: `position_embedding_type` `nope`) or a
    /// Mamba2 mixer (`layer_types`), then the shared SwiGLU MLP of `shared_intermediate_size`
    /// (the layer's dense MLP), plus, when `num_local_experts` > 0, the experts of
    /// `intermediate_size`, `num_experts_per_tok` per token, routed as `granitemoe` and added to
    /// the shared MLP without a gate. The multipliers as Granite 3.
    fn from_granite_hybrid(root: &Value, model_type: String) -> Result<Self> {
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let usize_of = |key: &str| -> Result<usize> {
            root.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        let f32_of = |key: &str, default: f64| {
            root.get(key).and_then(Value::as_f64).unwrap_or(default) as f32
        };
        let refuse = |what: &str| {
            Err(HerbertError::Config(format!(
                "config.json: granitemoehybrid {what} is not served"
            )))
        };
        if root.get("attention_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("with attention_bias");
        }
        if root.get("mamba_proj_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("with mamba_proj_bias");
        }
        if root.get("mamba_conv_bias").and_then(Value::as_bool) == Some(false) {
            return refuse("without mamba_conv_bias");
        }
        if root
            .get("time_step_limit")
            .and_then(Value::as_array)
            .is_some_and(|l| {
                l.first().and_then(Value::as_f64).unwrap_or(0.0) != 0.0
                    || l.get(1).and_then(Value::as_f64).is_some()
            })
        {
            return refuse("with a time_step_limit");
        }
        let hidden_size = usize_of("hidden_size")?;
        let num_attention_heads = usize_of("num_attention_heads")?;
        let head_dim = hidden_size / num_attention_heads.max(1);
        let num_layers = usize_of("num_hidden_layers")?;
        let layer_types: Vec<LayerType> = match root.get("layer_types").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|v| match v.as_str() {
                    Some("attention") => Ok(LayerType::FullAttention),
                    Some("mamba") => Ok(LayerType::LinearAttention),
                    other => Err(HerbertError::Config(format!(
                        "config.json: layer_types holds {other:?}"
                    ))),
                })
                .collect::<Result<_>>()?,
            None => vec![LayerType::FullAttention; num_layers],
        };
        let has_mamba = layer_types.contains(&LayerType::LinearAttention);
        let mamba = |key: &str| -> Result<usize> {
            if has_mamba {
                usize_of(key)
            } else {
                Ok(0)
            }
        };
        let experts = usize_of("num_local_experts").unwrap_or(0);
        let shared_inter = usize_of("shared_intermediate_size")?;
        let nope = root.get("position_embedding_type").and_then(Value::as_str) == Some("nope");
        if has_mamba
            && usize_of("mamba_expand")? * hidden_size
                != usize_of("mamba_n_heads")? * usize_of("mamba_d_head")?
        {
            return refuse("whose mamba_n_heads · mamba_d_head is not mamba_expand · hidden_size");
        }
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: shared_inter,
            rms_norm_eps: f32_of("rms_norm_eps", 1e-6),
            norm_type: NormType::RMSNorm,
            rope_theta: root
                .get("rope_theta")
                .or_else(|| {
                    root.get("rope_parameters")
                        .and_then(|r| r.get("rope_theta"))
                })
                .and_then(Value::as_f64)
                .unwrap_or(10000.0) as f32,
            rotary_ndims: head_dim,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: root
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            hidden_act: root
                .get("hidden_act")
                .and_then(Value::as_str)
                .unwrap_or("silu")
                .to_string(),
            attn_output_gate: false,
            qk_norm: false,
            qkv_bias: false,
            layer_types,
            linear_num_key_heads: mamba("mamba_n_groups")?,
            linear_num_value_heads: mamba("mamba_n_heads")?,
            linear_key_head_dim: mamba("mamba_d_state")?,
            linear_value_head_dim: mamba("mamba_d_head")?,
            linear_conv_kernel_dim: mamba("mamba_d_conv")?,
            gemma: None,
            qwen_moe: if experts > 0 {
                Some(QwenMoeConfig {
                    num_experts: experts,
                    top_k: usize_of("num_experts_per_tok")?,
                    inter: usize_of("intermediate_size")?,
                    shared_inter,
                    norm_topk_prob: true,
                    sparse: vec![true; num_layers],
                    dense_inter: 0,
                    lfm_scale: None,
                })
            } else {
                None
            },
            granite: Some(GraniteConfig {
                embedding_multiplier: f32_of("embedding_multiplier", 1.0),
                residual_multiplier: f32_of("residual_multiplier", 1.0),
                logits_scaling: f32_of("logits_scaling", 1.0),
                attention_multiplier: f32_of(
                    "attention_multiplier",
                    1.0 / (head_dim as f64).sqrt(),
                ),
            }),
            yarn: None,
            mamba: has_mamba,
            nope,
            no_rope: Vec::new(),
            lfm3: None,
            qwen_vl: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// LiquidAI's LFM2 and LFM2.5 (`lfm2`) and their mixtures of experts (`lfm2_moe`: LFM2-8B-A1B,
    /// LFM2-24B-A2B). Each layer is a gated short convolution (`conv`) or attention
    /// (`full_attention`: grouped-query, no bias, the heads RMS-normed by `w`, the whole head
    /// rotated); the norms are `w · rms(x)`; the final norm is the checkpoint's
    /// `embedding_norm`; the head is the embedding table. `lfm2`: a SwiGLU MLP of `block_ff_dim`,
    /// adjusted as `Lfm2MLP` does when `block_auto_adjust_ff_dim` (two thirds,
    /// `block_ffn_dim_multiplier`, rounded up to `block_multiple_of`). `lfm2_moe`: the first
    /// `num_dense_layers` layers have the SwiGLU MLP of `intermediate_size`, the others
    /// `num_experts` SwiGLU experts of `moe_intermediate_size`, `num_experts_per_tok`
    /// per token, LiquidAI's sigmoid route ([`QwenMoeConfig::lfm_scale`]), no shared expert. `lfm3`
    /// (LFM3-1.7B): LFM2's layers with a conv of `conv_L_cache` taps, `sliding_attention`
    /// layers besides the full ones, the gate per head and the sandwich norms ([`Lfm3Config`]).
    fn from_lfm2(root: &Value, model_type: String) -> Result<Self> {
        let missing = |key: &str| HerbertError::Config(format!("config.json: missing {key}"));
        let field = |key: &str| root.get(key);
        let usize_of = |key: &str| -> Result<usize> {
            field(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| missing(key))
        };
        let refuse = |what: &str| {
            Err(HerbertError::Config(format!(
                "config.json: {model_type} {what} is not served"
            )))
        };
        if field("conv_bias").and_then(Value::as_bool) == Some(true) {
            return refuse("with conv_bias");
        }
        if model_type == "lfm2_moe"
            && field("use_expert_bias").and_then(Value::as_bool) == Some(false)
        {
            return refuse("without use_expert_bias");
        }
        let hidden_size = usize_of("hidden_size")?;
        let num_attention_heads = usize_of("num_attention_heads")?;
        let head_dim = usize_of("head_dim").unwrap_or(hidden_size / num_attention_heads.max(1));
        let num_layers = usize_of("num_hidden_layers")?;
        let layer_types: Vec<LayerType> = match field("layer_types").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|v| match v.as_str() {
                    Some("full_attention") => Ok(LayerType::FullAttention),
                    Some("sliding_attention") if model_type == "lfm3" => {
                        Ok(LayerType::FullAttention)
                    }
                    Some("conv") => Ok(LayerType::ShortConv),
                    other => Err(HerbertError::Config(format!(
                        "config.json: layer_types holds {other:?}"
                    ))),
                })
                .collect::<Result<_>>()?,
            // The first LFM2 checkpoints list their attention layers instead.
            None => {
                let full: Vec<u64> = field("full_attn_idxs")
                    .and_then(Value::as_array)
                    .ok_or_else(|| missing("layer_types"))?
                    .iter()
                    .filter_map(Value::as_u64)
                    .collect();
                (0..num_layers as u64)
                    .map(|l| {
                        if full.contains(&l) {
                            LayerType::FullAttention
                        } else {
                            LayerType::ShortConv
                        }
                    })
                    .collect()
            }
        };
        let moe = model_type == "lfm2_moe";
        let mut inter = field("block_ff_dim")
            .or_else(|| field("intermediate_size"))
            .and_then(Value::as_u64)
            .ok_or_else(|| missing("block_ff_dim"))? as usize;
        if field("block_auto_adjust_ff_dim")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            inter = 2 * inter / 3;
            if let Some(m) = field("block_ffn_dim_multiplier").and_then(Value::as_f64) {
                inter = (m * inter as f64) as usize;
            }
            let multiple = usize_of("block_multiple_of")?.max(1);
            inter = multiple * inter.div_ceil(multiple);
        }
        let qwen_moe = if moe {
            let dense = usize_of("num_dense_layers").unwrap_or(0);
            Some(QwenMoeConfig {
                num_experts: usize_of("num_experts")?,
                top_k: usize_of("num_experts_per_tok")?,
                inter: usize_of("moe_intermediate_size")?,
                shared_inter: 0,
                norm_topk_prob: field("norm_topk_prob")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                sparse: (0..num_layers).map(|l| l >= dense).collect(),
                dense_inter: if dense > 0 { inter } else { 0 },
                lfm_scale: Some(
                    field("routed_scaling_factor")
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32,
                ),
            })
        } else {
            None
        };
        // LFM3: `rope_parameters` per layer type (`full_attention`, `sliding_attention`).
        let rp = field("rope_parameters");
        let full_rp = rp.and_then(|r| r.get("full_attention")).or(rp);
        let rope_theta = full_rp
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| field("rope_theta"))
            .and_then(Value::as_f64)
            .ok_or_else(|| missing("rope_theta"))? as f32;
        let lfm3 = if model_type == "lfm3" {
            let rope_type = |r: Option<&Value>| {
                r.and_then(|r| r.get("rope_type"))
                    .and_then(Value::as_str)
                    .unwrap_or("default")
                    .to_string()
            };
            let sliding_rp = rp.and_then(|r| r.get("sliding_attention"));
            if rope_type(full_rp) != "default" || rope_type(sliding_rp) != "default" {
                return refuse("with a RoPE other than the default one");
            }
            if sliding_rp
                .and_then(|r| r.get("partial_rotary_factor"))
                .and_then(Value::as_f64)
                .is_some_and(|f| f != 1.0)
            {
                return refuse("with a partial RoPE on its sliding layers");
            }
            let partial = full_rp
                .and_then(|r| r.get("partial_rotary_factor"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            let names: Vec<&str> = field("layer_types")
                .and_then(Value::as_array)
                .map(|l| l.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let sliding: Vec<bool> = (0..num_layers)
                .map(|l| names.get(l) == Some(&"sliding_attention"))
                .collect();
            let sliding_window =
                field("sliding_window").and_then(Value::as_u64).unwrap_or(0) as usize;
            if sliding.contains(&true) && sliding_window == 0 {
                return refuse("with sliding_attention layers and no sliding_window");
            }
            Some(Lfm3Config {
                sliding,
                sliding_window,
                rope_theta_sliding: sliding_rp
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(Value::as_f64)
                    .unwrap_or(rope_theta as f64) as f32,
                rope_angles: (partial * head_dim as f64) as usize / 2,
                sandwich_norm: field("use_sandwich_norm")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            })
        } else {
            None
        };
        let config = Config {
            model_type,
            vocab_size: usize_of("vocab_size")?,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim,
            intermediate_size: match &qwen_moe {
                Some(m) => m.dense_inter,
                None => inter,
            },
            rms_norm_eps: field("norm_eps")
                .or_else(|| field("block_norm_eps"))
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            norm_type: NormType::RMSNorm,
            rope_theta,
            rotary_ndims: head_dim,
            max_position_embeddings: usize_of("max_position_embeddings")?,
            tie_word_embeddings: field("tie_embedding")
                .or_else(|| field("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
            hidden_act: "silu".to_string(),
            attn_output_gate: false,
            qk_norm: true,
            qkv_bias: false,
            layer_types,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: usize_of("conv_L_cache")?,
            gemma: None,
            qwen_moe,
            granite: None,
            yarn: None,
            mamba: false,
            nope: false,
            no_rope: Vec::new(),
            lfm3,
            qwen_vl: None,
        };
        config.validate()?;
        Ok(config)
    }

    /// Is layer `l` a gated short convolution (`lfm2`, `lfm2_moe`)?
    pub fn is_short_conv_layer(&self, l: usize) -> bool {
        self.layer_types.get(l) == Some(&LayerType::ShortConv)
    }

    /// Granite 4 with experts: its shared MLP is the expert block's shared expert, added without a
    /// gate.
    pub fn mamba_or_hybrid_moe(&self) -> bool {
        self.model_type == "granitemoehybrid" && self.qwen_moe.is_some()
    }

    /// Is layer `l` a GatedDeltaNet layer?
    pub fn is_deltanet_layer(&self, l: usize) -> bool {
        self.layer_types.get(l) == Some(&LayerType::LinearAttention)
    }

    /// Has layer `l` an expert block (`qwen3_5_moe`)?
    pub fn is_moe_layer(&self, l: usize) -> bool {
        self.qwen_moe
            .as_ref()
            .is_some_and(|m| m.sparse.get(l) == Some(&true))
    }

    /// The width of the dense MLP of layer `l`: the shared expert's on a sparse layer of
    /// `qwen3_5_moe`.
    pub fn mlp_width(&self, l: usize) -> usize {
        match &self.qwen_moe {
            Some(m) if self.is_moe_layer(l) => m.shared_inter,
            Some(m) => m.dense_inter,
            None => self.intermediate_size,
        }
    }

    /// Rows of `q_proj` without the gate: `num_attention_heads · head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Rows of `k_proj` and `v_proj`: `num_key_value_heads · head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn validate(&self) -> Result<()> {
        let refuse = |what: &str| Err(HerbertError::Config(format!("config.json: {what}")));
        // A model whose every layer is an expert block without a shared expert (Granite 3) has no
        // dense MLP.
        let no_dense = self
            .qwen_moe
            .as_ref()
            .is_some_and(|m| m.shared_inter == 0 && m.dense_inter == 0);
        if self.hidden_size == 0
            || self.num_layers == 0
            || self.vocab_size == 0
            || (self.intermediate_size == 0 && !no_dense)
        {
            return refuse("a dimension is zero");
        }
        if self.layer_types.len() != self.num_layers {
            return refuse("layer_types does not have one entry per layer");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return refuse(
                "num_attention_heads must be a positive multiple of num_key_value_heads",
            );
        }
        if self.head_dim == 0 || self.rotary_ndims == 0 || self.rotary_ndims > self.head_dim {
            return refuse("rotary_ndims must lie in 1..=head_dim");
        }
        if self.layer_types.contains(&LayerType::LinearAttention)
            && (self.linear_num_key_heads == 0
                || self.linear_num_value_heads == 0
                || !self
                    .linear_num_value_heads
                    .is_multiple_of(self.linear_num_key_heads)
                || self.linear_key_head_dim == 0
                || self.linear_value_head_dim == 0
                || self.linear_conv_kernel_dim == 0)
        {
            return refuse(
                "the linear_* fields of the GatedDeltaNet mixer are missing or inconsistent",
            );
        }
        if self.layer_types.contains(&LayerType::ShortConv)
            && (self.linear_conv_kernel_dim < 2 || !self.hidden_size.is_multiple_of(16))
        {
            return refuse(
                "the short convolution wants conv_L_cache ≥ 2 and hidden_size a multiple of 16",
            );
        }
        if let Some(m) = &self.qwen_moe {
            if m.num_experts == 0
                || !m.num_experts.is_multiple_of(16)
                || m.top_k == 0
                || m.top_k > 8
                || m.top_k > m.num_experts
            {
                return refuse("mixture of experts: num_experts must be a positive multiple of 16 and num_experts_per_tok lie in 1..=8");
            }
            if m.inter == 0
                || !m.inter.is_multiple_of(32)
                || (m.shared_inter == 0 && self.model_type.starts_with("qwen3_5"))
                || !m.shared_inter.is_multiple_of(16)
            {
                return refuse("mixture of experts: moe_intermediate_size must be a multiple of 32, shared_expert_intermediate_size of 16");
            }
        }
        if let Some(v) = &self.qwen_vl {
            if v.mrope_section.iter().sum::<usize>() != self.rotary_ndims / 2
                || (v.interleaved
                    && (v.mrope_section[1] > v.mrope_section[0]
                        || v.mrope_section[2] > v.mrope_section[0]))
            {
                return refuse("qwen3_vl_moe: mrope_section must sum to head_dim / 2 (interleaved: h and w no wider than t)");
            }
        }
        Ok(())
    }
}
