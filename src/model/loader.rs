//! The checkpoint read into the model structures: every family but Gemma 4 (`model/gemma.rs`),
//! the text model only.
//!
//! What enters: the model directory (`config.json`, one or more
//! `*.safetensors` shards, their index when sharded) and a packer: the function
//! that turns a row-major bf16 matrix into the weight the engine reads.
//! What comes out: the `Config` and a `Model` — the embedding table, the
//! decoder layers (a mixer: attention, GatedDeltaNet, short convolution or Mamba2; then a
//! SwiGLU MLP or a mixture of experts), the final norm, the language head (the embedding
//! table again when it is tied) and the RoPE tables.
//!
//! Every matrix is mapped from the shard as bf16 and handed to the packer, which keeps it
//! bf16 or quantizes it (`--int8`, `--q4`, when the shape allows). The small vectors (norms, the
//! DeltaNet gates, decay and bias) are read as f32 or bf16 exactly as the reference implementation
//! keeps them. Tensors the engine does not use — the multi-token-prediction head `mtp.*`, the
//! vision tower — are skipped and counted.
//!
//! `qwen3_5_moe`: the sparse layers have `mlp.shared_expert.*` (read as the
//! dense MLP), `mlp.shared_expert_gate`, the router `mlp.gate` (kept bf16 in
//! blocks of sixteen experts) and the experts stacked in two tensors,
//! `mlp.experts.gate_up_proj` `[experts, 2·inter, hidden]` (gate rows then up
//! rows) and `mlp.experts.down_proj` `[experts, hidden, inter]`, packed one
//! expert at a time.

use crate::core::avec::AVec;
use crate::core::config::{Config, QwenVlConfig};
use crate::core::error::{HerbertError, Result};
use crate::core::tensor::{bf16_to_f32, f32_to_bf16, BF16};
use crate::model::model::{
    Attention, DecoderLayer, GatedDeltaNet, LayerBlock, Mlp, Model, QwenMoe,
};
use crate::model::weight::PackedBf16R16Weight;
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A Hugging Face cache directory points at a snapshot; a plain directory is
/// used as is.
pub fn resolve_model_dir(model_dir: &Path) -> Result<PathBuf> {
    if model_dir.join("config.json").exists() {
        return Ok(model_dir.to_path_buf());
    }
    let snapshots = model_dir.join("snapshots");
    if !snapshots.is_dir() {
        return Ok(model_dir.to_path_buf());
    }
    if let Ok(reference) = std::fs::read_to_string(model_dir.join("refs").join("main")) {
        let candidate = snapshots.join(reference.trim());
        if candidate.join("config.json").exists() {
            return Ok(candidate);
        }
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.join("config.json").exists())
        .collect();
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        HerbertError::ModelLoad(format!(
            "no snapshot with a config.json under {}",
            snapshots.display()
        ))
    })
}

/// A bf16 vector from a tensor view; f32 and f16 are rounded to nearest even.
pub fn load_bf16_from_view(view: &TensorView<'_>, name: &str) -> Result<Vec<BF16>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32_to_bf16(half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()))
            .collect()),
        safetensors::Dtype::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect()),
        dtype => Err(HerbertError::ModelLoad(format!(
            "unsupported dtype {dtype:?} for {name}"
        ))),
    }
}

/// An f32 vector from a tensor view; bf16 and f16 are widened exactly.
pub fn load_f32_from_view(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        safetensors::Dtype::BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect()),
        dtype => Err(HerbertError::ModelLoad(format!(
            "unsupported dtype {dtype:?} for {name}"
        ))),
    }
}

/// The RoPE tables, `[max_position_embeddings, rotary_ndims / 2]` each:
/// `cos(pos · θ^(−2i/rotary_ndims))` and its sine. Qwen3.5 declares an MRoPE
/// (three position axes for images); on text the three axes carry the same
/// position and the interleaved pattern is inert, so the tables are the
/// ordinary ones.
pub fn compute_rope_cache(config: &Config) -> (Vec<f32>, Vec<f32>) {
    let rotary_ndims = config.rotary_ndims;
    let half_dim = rotary_ndims / 2;
    let positions = config.max_position_embeddings;
    let inv_freq: Vec<f32> = match &config.yarn {
        // No position (Granite 4 H): every angle is zero, cos 1 and sin 0, the rotation the
        // identity.
        _ if config.nope => vec![0.0; half_dim],
        // LFM3's full layers: the first `rope_angles` pairs at the frequencies of the whole head,
        // the others by zero (`Lfm3RotaryEmbedding`, the zero-tail partial RoPE).
        None if config.lfm3.is_some() => {
            let angles = config
                .lfm3
                .as_ref()
                .map_or(half_dim, |c| c.rope_angles.min(half_dim));
            (0..half_dim)
                .map(|i| {
                    if i < angles {
                        1.0 / config.rope_theta.powf((2 * i) as f32 / rotary_ndims as f32)
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        Some(y) => y.inv_freq(config.rope_theta, rotary_ndims),
        None => (0..half_dim)
            .map(|i| 1.0 / config.rope_theta.powf((2 * i) as f32 / rotary_ndims as f32))
            .collect(),
    };
    let scale = config.yarn.map_or(1.0, |y| y.attention_factor as f32);
    let mut cos = vec![0.0f32; positions * half_dim];
    let mut sin = vec![0.0f32; positions * half_dim];
    // Qwen3-VL: the row of a text token, `t = h = w = pos`, by the multimodal RoPE's own function.
    if let Some(vl) = &config.qwen_vl {
        for pos in 0..positions {
            let (c, s) = mrope_row(vl, &inv_freq, [pos; 3]);
            cos[pos * half_dim..(pos + 1) * half_dim].copy_from_slice(&c);
            sin[pos * half_dim..(pos + 1) * half_dim].copy_from_slice(&s);
        }
        return (cos, sin);
    }
    for pos in 0..positions {
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freq[i];
            cos[pos * half_dim + i] = angle.cos() * scale;
            sin[pos * half_dim + i] = angle.sin() * scale;
        }
    }
    (cos, sin)
}

/// The inverse frequencies of the plain RoPE of `config` (`1 / θ^(2i / rotary_ndims)`), the ones
/// the tables use.
pub fn rope_inv_freq(config: &Config) -> Vec<f32> {
    let n = config.rotary_ndims;
    (0..n / 2)
        .map(|i| 1.0 / config.rope_theta.powf((2 * i) as f32 / n as f32))
        .collect()
}

/// One row of the multimodal RoPE of Qwen3-VL (`Qwen3VLMoeTextRotaryEmbedding`): frequency `i`
/// turns by the position of its axis ([`QwenVlConfig::mrope_axis`]), `angle = position[axis] ·
/// inv_freq[i]` in f32 as torch does, then its cos and sin (half a head each; the kernels repeat
/// the half). A text token (`t = h = w`) gets the plain RoPE's row; an image token its `(t, h, w)`.
/// The entry point of the vision of Qwen3-VL: the rows of a prompt with an image are this function
/// of each token's three positions.
pub fn mrope_row(
    vl: &QwenVlConfig,
    inv_freq: &[f32],
    position: [usize; 3],
) -> (Vec<f32>, Vec<f32>) {
    inv_freq
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let angle = position[vl.mrope_axis(i)] as f32 * f;
            (angle.cos(), angle.sin())
        })
        .unzip()
}

/// LFM3's sliding layers: `cos(pos · θ_s^(−2i/head_dim))` and its sine over the whole head, `θ_s`
/// their own base.
pub fn compute_rope_sliding(
    config: &Config,
    c: &crate::core::config::Lfm3Config,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = config.rotary_ndims / 2;
    let positions = config.max_position_embeddings;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            1.0 / c
                .rope_theta_sliding
                .powf((2 * i) as f32 / config.rotary_ndims as f32)
        })
        .collect();
    let mut cos = vec![0.0f32; positions * half_dim];
    let mut sin = vec![0.0f32; positions * half_dim];
    for pos in 0..positions {
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freq[i];
            cos[pos * half_dim + i] = angle.cos();
            sin[pos * half_dim + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// What a tensor name means to this model.
#[derive(Debug)]
enum Kind {
    Embedding,
    FinalNorm,
    LmHead,
    /// A layer's norm or small vector, kept bf16.
    Bf16 {
        layer: usize,
        field: &'static str,
    },
    /// A layer's small vector, kept f32 (the DeltaNet gates, conv, decay, bias).
    F32 {
        layer: usize,
        field: &'static str,
    },
    /// A layer's matrix, packed for the GEMM.
    Matrix {
        layer: usize,
        field: &'static str,
    },
    /// `qwen3_5_moe`: the router of a layer, `[experts, hidden]`.
    Router {
        layer: usize,
    },
    /// `qwen3_5_moe`: the stacked experts of a layer, `[experts, rows, columns]`.
    Experts {
        layer: usize,
        field: &'static str,
    },
    /// Granite 4: `mamba.in_proj`, `[z; x B C; dt]`, split into three.
    MambaInProj {
        layer: usize,
    },
    /// Granite 4: `shared_mlp.input_linear`, `[gate; up]`, split into two.
    SharedGateUp {
        layer: usize,
    },
    /// LFM2: `conv.in_proj`, `[B; C; x]`, split into three.
    ShortConvInProj {
        layer: usize,
    },
    /// `lfm2_moe`: one matrix of one expert (`feed_forward.experts.E.w1` gate, `w3` up, `w2`
    /// down).
    Expert {
        layer: usize,
        expert: usize,
        field: &'static str,
    },
    Skipped,
}

/// The tensor names of a Qwen3.5 text checkpoint. The prefix varies with the
/// packaging (`model.`, `model.language_model.`, `language_model.model.`).
fn classify(name: &str) -> Kind {
    // Ministral 3 (`mistral3`) keeps its head under the text model's prefix.
    if name == "language_model.lm_head.weight" {
        return Kind::LmHead;
    }
    let name = name
        .strip_prefix("language_model.model.")
        .or_else(|| name.strip_prefix("model.language_model."))
        .or_else(|| name.strip_prefix("model."))
        // Qwen3-Embedding: the text model's names without a prefix (`layers.0.…`, `embed_tokens`,
        // `norm`).
        .or_else(|| {
            (name.starts_with("layers.") || name == "embed_tokens.weight" || name == "norm.weight")
                .then_some(name)
        })
        .map(|rest| format!("model.{rest}"))
        .unwrap_or_else(|| name.to_string());
    match name.as_str() {
        "model.embed_tokens.weight" => return Kind::Embedding,
        "model.norm.weight" => return Kind::FinalNorm,
        // LFM2: the final norm is named after the embedding.
        "model.embedding_norm.weight" => return Kind::FinalNorm,
        "lm_head.weight" => return Kind::LmHead,
        _ => {}
    }
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return Kind::Skipped;
    };
    let Some((index, field)) = rest.split_once('.') else {
        return Kind::Skipped;
    };
    let Ok(layer) = index.parse::<usize>() else {
        return Kind::Skipped;
    };
    match field {
        "input_layernorm.weight" => Kind::Bf16 {
            layer,
            field: "input_layernorm",
        },
        "post_attention_layernorm.weight" => Kind::Bf16 {
            layer,
            field: "post_attention_layernorm",
        },
        "self_attn.q_proj.weight" => Kind::Matrix {
            layer,
            field: "q_proj",
        },
        "self_attn.k_proj.weight" => Kind::Matrix {
            layer,
            field: "k_proj",
        },
        "self_attn.v_proj.weight" => Kind::Matrix {
            layer,
            field: "v_proj",
        },
        "self_attn.o_proj.weight" => Kind::Matrix {
            layer,
            field: "o_proj",
        },
        "self_attn.q_norm.weight" => Kind::Bf16 {
            layer,
            field: "q_norm",
        },
        "self_attn.k_norm.weight" => Kind::Bf16 {
            layer,
            field: "k_norm",
        },
        // Qwen2.5.
        "self_attn.q_proj.bias" => Kind::Bf16 {
            layer,
            field: "q_bias",
        },
        "self_attn.k_proj.bias" => Kind::Bf16 {
            layer,
            field: "k_bias",
        },
        "self_attn.v_proj.bias" => Kind::Bf16 {
            layer,
            field: "v_bias",
        },
        "mlp.gate_proj.weight" => Kind::Matrix {
            layer,
            field: "gate_proj",
        },
        "mlp.up_proj.weight" => Kind::Matrix {
            layer,
            field: "up_proj",
        },
        "mlp.down_proj.weight" => Kind::Matrix {
            layer,
            field: "down_proj",
        },
        // `qwen3_5_moe`: the shared expert is the layer's dense MLP.
        "mlp.shared_expert.gate_proj.weight" => Kind::Matrix {
            layer,
            field: "gate_proj",
        },
        "mlp.shared_expert.up_proj.weight" => Kind::Matrix {
            layer,
            field: "up_proj",
        },
        "mlp.shared_expert.down_proj.weight" => Kind::Matrix {
            layer,
            field: "down_proj",
        },
        "mlp.shared_expert_gate.weight" => Kind::Bf16 {
            layer,
            field: "shared_expert_gate",
        },
        "mlp.gate.weight" => Kind::Router { layer },
        "mlp.experts.gate_up_proj" => Kind::Experts {
            layer,
            field: "gate_up",
        },
        "mlp.experts.down_proj" => Kind::Experts {
            layer,
            field: "down",
        },
        // Granite 3 (`granitemoe`): the router and the stacked experts, `input_linear` `[experts,
        // 2·inter, hidden]` (gate rows then up rows, as `chunk(2)` reads them) and
        // `output_linear` `[experts, hidden, inter]`.
        "block_sparse_moe.router.layer.weight" => Kind::Router { layer },
        "block_sparse_moe.input_linear.weight" => Kind::Experts {
            layer,
            field: "gate_up",
        },
        "block_sparse_moe.output_linear.weight" => Kind::Experts {
            layer,
            field: "down",
        },
        "linear_attn.in_proj_qkv.weight" => Kind::Matrix {
            layer,
            field: "in_proj_qkv",
        },
        "linear_attn.in_proj_z.weight" => Kind::Matrix {
            layer,
            field: "in_proj_z",
        },
        "linear_attn.out_proj.weight" => Kind::Matrix {
            layer,
            field: "out_proj",
        },
        "linear_attn.in_proj_a.weight" => Kind::F32 {
            layer,
            field: "in_proj_a",
        },
        "linear_attn.in_proj_b.weight" => Kind::F32 {
            layer,
            field: "in_proj_b",
        },
        "linear_attn.conv1d.weight" => Kind::F32 {
            layer,
            field: "conv1d",
        },
        "linear_attn.A_log" => Kind::F32 {
            layer,
            field: "a_log",
        },
        "linear_attn.dt_bias" => Kind::F32 {
            layer,
            field: "dt_bias",
        },
        "linear_attn.norm.weight" => Kind::Bf16 {
            layer,
            field: "norm",
        },
        // Granite 4 (`granitemoehybrid`): Mamba2, whose `in_proj` the loader splits, and the shared
        // MLP, whose `input_linear` is `[gate; up]`.
        "mamba.in_proj.weight" => Kind::MambaInProj { layer },
        "mamba.conv1d.weight" => Kind::F32 {
            layer,
            field: "conv1d",
        },
        "mamba.conv1d.bias" => Kind::F32 {
            layer,
            field: "conv1d_bias",
        },
        "mamba.A_log" => Kind::F32 {
            layer,
            field: "a_log",
        },
        "mamba.dt_bias" => Kind::F32 {
            layer,
            field: "dt_bias",
        },
        "mamba.D" => Kind::F32 { layer, field: "d" },
        "mamba.norm.weight" => Kind::Bf16 {
            layer,
            field: "norm",
        },
        "mamba.out_proj.weight" => Kind::Matrix {
            layer,
            field: "out_proj",
        },
        "shared_mlp.input_linear.weight" => Kind::SharedGateUp { layer },
        "shared_mlp.output_linear.weight" => Kind::Matrix {
            layer,
            field: "down_proj",
        },
        // LFM2 and `lfm2_moe`: the norms, the attention, the gated short convolution, the MLP or
        // the experts.
        "operator_norm.weight" => Kind::Bf16 {
            layer,
            field: "input_layernorm",
        },
        "ffn_norm.weight" => Kind::Bf16 {
            layer,
            field: "post_attention_layernorm",
        },
        "self_attn.out_proj.weight" => Kind::Matrix {
            layer,
            field: "o_proj",
        },
        "self_attn.q_layernorm.weight" => Kind::Bf16 {
            layer,
            field: "q_norm",
        },
        "self_attn.k_layernorm.weight" => Kind::Bf16 {
            layer,
            field: "k_norm",
        },
        // LFM3: the gate per head, the sandwich norms.
        "self_attn.gate_proj.weight" => Kind::Matrix {
            layer,
            field: "head_gate",
        },
        "post_operator_norm.weight" => Kind::Bf16 {
            layer,
            field: "post_operator_norm",
        },
        "post_ffn_norm.weight" => Kind::Bf16 {
            layer,
            field: "post_ffn_norm",
        },
        "conv.in_proj.weight" => Kind::ShortConvInProj { layer },
        "conv.conv.weight" => Kind::F32 {
            layer,
            field: "conv1d",
        },
        "conv.out_proj.weight" => Kind::Matrix {
            layer,
            field: "out_proj",
        },
        "feed_forward.w1.weight" => Kind::Matrix {
            layer,
            field: "gate_proj",
        },
        "feed_forward.w3.weight" => Kind::Matrix {
            layer,
            field: "up_proj",
        },
        "feed_forward.w2.weight" => Kind::Matrix {
            layer,
            field: "down_proj",
        },
        "feed_forward.gate.weight" => Kind::Router { layer },
        "feed_forward.expert_bias" => Kind::F32 {
            layer,
            field: "expert_bias",
        },
        // `qwen3_moe` (Qwen3-30B-A3B): one matrix per expert, `[gate; up]` put end to end as
        // transformers' `qwen2_moe` conversion does (`MergeModulelist`,
        // `Concatenate(dim=1)`).
        _ if field.starts_with("mlp.experts.") => {
            match field["mlp.experts.".len()..].split_once('.') {
                Some((e, w)) => match (e.parse::<usize>(), w) {
                    (Ok(expert), "gate_proj.weight") => Kind::Expert {
                        layer,
                        expert,
                        field: "gate",
                    },
                    (Ok(expert), "up_proj.weight") => Kind::Expert {
                        layer,
                        expert,
                        field: "up",
                    },
                    (Ok(expert), "down_proj.weight") => Kind::Expert {
                        layer,
                        expert,
                        field: "down",
                    },
                    _ => Kind::Skipped,
                },
                None => Kind::Skipped,
            }
        }
        _ => match field
            .strip_prefix("feed_forward.experts.")
            .and_then(|r| r.split_once('.'))
        {
            Some((e, w)) => match (e.parse::<usize>(), w) {
                (Ok(expert), "w1.weight") => Kind::Expert {
                    layer,
                    expert,
                    field: "gate",
                },
                (Ok(expert), "w3.weight") => Kind::Expert {
                    layer,
                    expert,
                    field: "up",
                },
                (Ok(expert), "w2.weight") => Kind::Expert {
                    layer,
                    expert,
                    field: "down",
                },
                _ => Kind::Skipped,
            },
            None => Kind::Skipped,
        },
    }
}

/// `w ← bf16(w · s)` (or `bf16(w / s)`), each product rounded to nearest even once: a Granite
/// multiplier folded into a matrix at load.
fn fold(data: &mut [BF16], s: f32, divide: bool) {
    for w in data {
        let x = bf16_to_f32(*w);
        *w = f32_to_bf16(if divide { x / s } else { x * s });
    }
}

/// The conv channel the reordered channel `c` comes from: Mamba2's `[x | B | C]` (`x` `value` wide,
/// `B` and `C` `key` wide) becomes `[C | B | x]`, the DeltaNet's `[q | k | v]`.
fn mamba_channel(c: usize, key: usize, value: usize) -> usize {
    if c < key {
        value + key + c
    } else if c < 2 * key {
        value + c - key
    } else {
        c - 2 * key
    }
}

/// One tensor, read and converted, before it joins its layer.
enum Loaded {
    Embedding(Vec<BF16>),
    FinalNorm(Vec<BF16>),
    LmHead(Vec<BF16>),
    Bf16 {
        layer: usize,
        field: &'static str,
        data: Vec<BF16>,
    },
    F32 {
        layer: usize,
        field: &'static str,
        data: Vec<f32>,
    },
    Matrix {
        layer: usize,
        field: &'static str,
        weight: PackedBf16R16Weight,
    },
    Router {
        layer: usize,
        blocks: Vec<PackedBf16R16Weight>,
    },
    Experts {
        layer: usize,
        field: &'static str,
        weights: Vec<PackedBf16R16Weight>,
    },
    MambaInProj {
        layer: usize,
        z: PackedBf16R16Weight,
        xbc: PackedBf16R16Weight,
        dt: Vec<f32>,
    },
    SharedGateUp {
        layer: usize,
        gate: PackedBf16R16Weight,
        up: PackedBf16R16Weight,
    },
    ShortConvInProj {
        layer: usize,
        b: PackedBf16R16Weight,
        c: PackedBf16R16Weight,
        x: PackedBf16R16Weight,
    },
    Expert {
        layer: usize,
        expert: usize,
        field: &'static str,
        weight: PackedBf16R16Weight,
    },
    Skipped(String),
}

/// The tensors of one layer, collected across shards, then assembled.
#[derive(Default)]
struct PartialLayer {
    bf16: HashMap<&'static str, Vec<BF16>>,
    f32: HashMap<&'static str, Vec<f32>>,
    matrices: HashMap<&'static str, PackedBf16R16Weight>,
    router: Vec<PackedBf16R16Weight>,
    experts: HashMap<&'static str, Vec<PackedBf16R16Weight>>,
    /// `lfm2_moe`: the experts one matrix at a time, `(field, expert)`.
    expert: HashMap<(&'static str, usize), PackedBf16R16Weight>,
}

impl PartialLayer {
    /// The vectors of a layer (norms, DeltaNet parameters), each on a cache line.
    fn take_bf16(&mut self, layer: usize, field: &str) -> Result<AVec<BF16>> {
        self.bf16
            .remove(field)
            .map(AVec::from_vec)
            .ok_or_else(|| HerbertError::ModelLoad(format!("layer {layer}: missing {field}")))
    }
    fn take_f32(&mut self, layer: usize, field: &str) -> Result<AVec<f32>> {
        self.f32
            .remove(field)
            .map(AVec::from_vec)
            .ok_or_else(|| HerbertError::ModelLoad(format!("layer {layer}: missing {field}")))
    }
    fn take_matrix(&mut self, layer: usize, field: &str) -> Result<PackedBf16R16Weight> {
        self.matrices
            .remove(field)
            .ok_or_else(|| HerbertError::ModelLoad(format!("layer {layer}: missing {field}")))
    }

    /// The decoder layer: its mixer (attention or GatedDeltaNet, decided by the
    /// config's layer types) and its dense SwiGLU MLP.
    fn finalize(mut self, layer: usize, config: &Config) -> Result<DecoderLayer> {
        let block = if config.is_short_conv_layer(layer) {
            let (h, ck) = (config.hidden_size, config.linear_conv_kernel_dim);
            let conv1d_weight = self.take_f32(layer, "conv1d")?;
            if conv1d_weight.len() != h * ck {
                return Err(HerbertError::ModelLoad(format!(
                    "layer {layer}: conv.conv holds {} weights, not {h} × {ck}",
                    conv1d_weight.len()
                )));
            }
            let empty = || AVec::from_vec(Vec::new());
            LayerBlock::GatedDeltaNet(GatedDeltaNet {
                in_proj_qkv: self.take_matrix(layer, "in_proj_qkv")?,
                in_proj_z: self.take_matrix(layer, "in_proj_z")?,
                in_proj_a: empty(),
                in_proj_b: empty(),
                conv1d_weight,
                a_log: empty(),
                dt_bias: empty(),
                norm_weight: AVec::from_vec(Vec::new()),
                out_proj: self.take_matrix(layer, "out_proj")?,
                num_k_heads: 0,
                num_v_heads: 0,
                head_k_dim: 0,
                head_v_dim: h,
                conv_kernel_size: ck,
                mamba: false,
                conv1d_bias: empty(),
                d: empty(),
                short_conv: true,
                in_proj_c: self.take_matrix(layer, "in_proj_c")?,
            })
        } else if config.is_deltanet_layer(layer) && config.mamba {
            let (key, value, nv, ck) = (
                config.linear_num_key_heads * config.linear_key_head_dim,
                config.linear_num_value_heads * config.linear_value_head_dim,
                config.linear_num_value_heads,
                config.linear_conv_kernel_dim,
            );
            let conv = 2 * key + value;
            // The checkpoint's conv channels, `[x | B | C]`, put in the order of `in_proj_qkv`.
            let w = self.take_f32(layer, "conv1d")?;
            let bias = self.take_f32(layer, "conv1d_bias")?;
            if w.len() != conv * ck || bias.len() != conv {
                return Err(HerbertError::ModelLoad(format!(
                    "layer {layer}: mamba.conv1d of {} weights and {} biases",
                    w.len(),
                    bias.len()
                )));
            }
            let conv1d_weight: Vec<f32> = (0..conv)
                .flat_map(|c| {
                    let o = mamba_channel(c, key, value);
                    w[o * ck..(o + 1) * ck].to_vec()
                })
                .collect();
            let conv1d_bias: Vec<f32> = (0..conv)
                .map(|c| bias[mamba_channel(c, key, value)])
                .collect();
            let norm_weight = self.take_bf16(layer, "norm")?;
            if norm_weight.len() != value {
                return Err(HerbertError::ModelLoad(format!(
                    "layer {layer}: mamba.norm holds {} values, not {value}",
                    norm_weight.len()
                )));
            }
            LayerBlock::GatedDeltaNet(GatedDeltaNet {
                in_proj_qkv: self.take_matrix(layer, "in_proj_qkv")?,
                in_proj_z: self.take_matrix(layer, "in_proj_z")?,
                in_proj_a: self.take_f32(layer, "in_proj_a")?,
                in_proj_b: AVec::from_vec(vec![0.0; nv * config.hidden_size]),
                conv1d_weight: AVec::from_vec(conv1d_weight),
                a_log: self.take_f32(layer, "a_log")?,
                dt_bias: self.take_f32(layer, "dt_bias")?,
                norm_weight,
                out_proj: self.take_matrix(layer, "out_proj")?,
                num_k_heads: config.linear_num_key_heads,
                num_v_heads: nv,
                head_k_dim: config.linear_key_head_dim,
                head_v_dim: config.linear_value_head_dim,
                conv_kernel_size: ck,
                mamba: true,
                conv1d_bias: AVec::from_vec(conv1d_bias),
                d: self.take_f32(layer, "d")?,
                short_conv: false,
                in_proj_c: PackedBf16R16Weight::empty(0, config.hidden_size),
            })
        } else if config.is_deltanet_layer(layer) {
            LayerBlock::GatedDeltaNet(GatedDeltaNet {
                in_proj_qkv: self.take_matrix(layer, "in_proj_qkv")?,
                in_proj_z: self.take_matrix(layer, "in_proj_z")?,
                in_proj_a: self.take_f32(layer, "in_proj_a")?,
                in_proj_b: self.take_f32(layer, "in_proj_b")?,
                conv1d_weight: self.take_f32(layer, "conv1d")?,
                a_log: self.take_f32(layer, "a_log")?,
                dt_bias: self.take_f32(layer, "dt_bias")?,
                norm_weight: self.take_bf16(layer, "norm")?,
                out_proj: self.take_matrix(layer, "out_proj")?,
                num_k_heads: config.linear_num_key_heads,
                num_v_heads: config.linear_num_value_heads,
                head_k_dim: config.linear_key_head_dim,
                head_v_dim: config.linear_value_head_dim,
                conv_kernel_size: config.linear_conv_kernel_dim,
                mamba: false,
                conv1d_bias: AVec::from_vec(Vec::new()),
                d: AVec::from_vec(Vec::new()),
                short_conv: false,
                in_proj_c: PackedBf16R16Weight::empty(0, config.hidden_size),
            })
        } else {
            LayerBlock::Attention(Attention {
                q_proj: self.take_matrix(layer, "q_proj")?,
                k_proj: self.take_matrix(layer, "k_proj")?,
                v_proj: self.take_matrix(layer, "v_proj")?,
                o_proj: self.take_matrix(layer, "o_proj")?,
                // Qwen2.5 has no per-head norms but biases on its three projections.
                q_norm: if config.qk_norm {
                    self.take_bf16(layer, "q_norm")?
                } else {
                    AVec::from_vec(Vec::new())
                },
                k_norm: if config.qk_norm {
                    self.take_bf16(layer, "k_norm")?
                } else {
                    AVec::from_vec(Vec::new())
                },
                q_bias: if config.qkv_bias {
                    self.take_bf16(layer, "q_bias")?
                } else {
                    AVec::from_vec(Vec::new())
                },
                k_bias: if config.qkv_bias {
                    self.take_bf16(layer, "k_bias")?
                } else {
                    AVec::from_vec(Vec::new())
                },
                v_bias: if config.qkv_bias {
                    self.take_bf16(layer, "v_bias")?
                } else {
                    AVec::from_vec(Vec::new())
                },
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                // SmolLM3: a layer without position turns its first pairs by zero
                // (`NOPE_ROTARY_NDIMS`).
                rotary_ndims: if config.no_rope.get(layer) == Some(&true) {
                    crate::model::model::NOPE_ROTARY_NDIMS
                } else {
                    config.rotary_ndims
                },
            })
        };
        // `qwen3_5_moe`: the experts of a sparse layer, whose dense MLP is the shared expert.
        let qwen_moe = match &config.qwen_moe {
            Some(m) if config.is_moe_layer(layer) => {
                // `lfm2_moe`: the experts one matrix at a time, `[gate; up]` put end to end as the
                // stacked checkpoints hold them.
                if !self.expert.is_empty() {
                    let mut gate_up = Vec::with_capacity(m.num_experts);
                    let mut down = Vec::with_capacity(m.num_experts);
                    for e in 0..m.num_experts {
                        let mut take = |f: &'static str| {
                            self.expert.remove(&(f, e)).ok_or_else(|| {
                                HerbertError::ModelLoad(format!(
                                    "layer {layer}: missing feed_forward.experts.{e} ({f})"
                                ))
                            })
                        };
                        let (gate, up) = (take("gate")?, take("up")?);
                        gate_up.push(PackedBf16R16Weight::stack(&gate, &up)?);
                        down.push(take("down")?);
                    }
                    self.experts.insert("gate_up", gate_up);
                    self.experts.insert("down", down);
                }
                let router = std::mem::take(&mut self.router);
                let gate_up = self.experts.remove("gate_up").ok_or_else(|| {
                    HerbertError::ModelLoad(format!(
                        "layer {layer}: missing mlp.experts.gate_up_proj"
                    ))
                })?;
                let down = self.experts.remove("down").ok_or_else(|| {
                    HerbertError::ModelLoad(format!("layer {layer}: missing mlp.experts.down_proj"))
                })?;
                if router.len() * 16 != m.num_experts
                    || gate_up.len() != m.num_experts
                    || down.len() != m.num_experts
                {
                    return Err(HerbertError::ModelLoad(format!(
                        "layer {layer}: the router or the experts do not count {} experts",
                        m.num_experts
                    )));
                }
                if gate_up
                    .iter()
                    .any(|w| w.n != 2 * m.inter || w.k != config.hidden_size)
                    || down
                        .iter()
                        .any(|w| w.n != config.hidden_size || w.k != m.inter)
                {
                    return Err(HerbertError::ModelLoad(format!(
                        "layer {layer}: experts of unexpected shape (moe_intermediate_size {})",
                        m.inter
                    )));
                }
                // Granite 3, `lfm2_moe`, `qwen3_moe` and Qwen3-VL have no shared expert: a gate of
                // zeros on an output of zeros adds exactly nothing.
                let shared_gate = if m.shared_inter == 0 {
                    AVec::from_vec(vec![0; config.hidden_size])
                } else {
                    self.take_bf16(layer, "shared_expert_gate")?
                };
                if shared_gate.len() != config.hidden_size {
                    return Err(HerbertError::ModelLoad(format!(
                        "layer {layer}: mlp.shared_expert_gate holds {} values",
                        shared_gate.len()
                    )));
                }
                let expert_bias = if m.lfm_scale.is_some() {
                    self.take_f32(layer, "expert_bias")?
                } else {
                    AVec::from_vec(Vec::new())
                };
                if m.lfm_scale.is_some() && expert_bias.len() != m.num_experts {
                    return Err(HerbertError::ModelLoad(format!(
                        "layer {layer}: feed_forward.expert_bias holds {} values",
                        expert_bias.len()
                    )));
                }
                Some(QwenMoe {
                    router,
                    shared_gate,
                    gate_up,
                    down,
                    expert_bias,
                })
            }
            _ => None,
        };
        let intermediate_size = config.mlp_width(layer);
        // No dense MLP (Granite 3): empty matrices of no rows, which the scheduler skips.
        if intermediate_size == 0 {
            for f in ["gate_proj", "up_proj"] {
                self.matrices
                    .insert(f, PackedBf16R16Weight::empty(0, config.hidden_size));
            }
            self.matrices.insert(
                "down_proj",
                PackedBf16R16Weight::empty(config.hidden_size, 0),
            );
        }
        let gate_proj = self.take_matrix(layer, "gate_proj")?;
        if gate_proj.n != intermediate_size {
            return Err(HerbertError::ModelLoad(format!(
                "layer {layer}: gate_proj has {} rows, expected {intermediate_size}",
                gate_proj.n
            )));
        }
        let lfm3 = match &config.lfm3 {
            Some(c) => {
                let empty = || AVec::from_vec(Vec::new());
                let (post_operator_norm, post_ffn_norm) = if c.sandwich_norm {
                    (
                        self.take_bf16(layer, "post_operator_norm")?,
                        self.take_bf16(layer, "post_ffn_norm")?,
                    )
                } else {
                    (empty(), empty())
                };
                if c.sandwich_norm
                    && (post_operator_norm.len() != config.hidden_size
                        || post_ffn_norm.len() != config.hidden_size)
                {
                    return Err(HerbertError::ModelLoad(format!(
                        "layer {layer}: sandwich norms of unexpected length"
                    )));
                }
                let head_gate = if config.is_short_conv_layer(layer) {
                    PackedBf16R16Weight::empty(0, config.hidden_size)
                } else {
                    let g = self.take_matrix(layer, "head_gate")?;
                    if g.n != config.num_attention_heads || g.k != config.hidden_size {
                        return Err(HerbertError::ModelLoad(format!(
                            "layer {layer}: self_attn.gate_proj is [{}, {}], not [{}, {}]",
                            g.n, g.k, config.num_attention_heads, config.hidden_size
                        )));
                    }
                    g
                };
                let window = if c.sliding.get(layer) == Some(&true) {
                    c.sliding_window
                } else {
                    0
                };
                Some(crate::model::model::Lfm3Layer {
                    post_operator_norm,
                    post_ffn_norm,
                    head_gate,
                    window,
                })
            }
            None => None,
        };
        Ok(DecoderLayer {
            lfm3,
            input_layernorm: self.take_bf16(layer, "input_layernorm")?,
            block,
            post_attention_layernorm: self.take_bf16(layer, "post_attention_layernorm")?,
            mlp: Mlp {
                gate_proj,
                up_proj: self.take_matrix(layer, "up_proj")?,
                down_proj: self.take_matrix(layer, "down_proj")?,
                intermediate_size,
            },
            gemma: None,
            qwen_moe,
        })
    }
}

/// Read the checkpoint. `pack(matrix, rows, columns, name)` turns each
/// row-major bf16 matrix into the engine's weight; it runs on several threads,
/// one slice of the shard's tensors each.
pub fn load_model(
    model_dir: &Path,
    pack: impl Fn(&[BF16], usize, usize, &str) -> Result<PackedBf16R16Weight> + Send + Sync,
    show_progress: bool,
) -> Result<(Config, Model)> {
    let model_dir = resolve_model_dir(model_dir)?;
    let config = Config::from_file(&model_dir.join("config.json"))?;
    if config.gemma.is_some() {
        return crate::model::gemma::load_model(&model_dir, pack, show_progress);
    }

    // The shards, and which tensor lives in which one.
    let index_path = model_dir.join("model.safetensors.index.json");
    let (shards, tensor_shard): (Vec<String>, Option<HashMap<String, String>>) =
        if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&std::fs::read(&index_path)?)?;
            let names: HashSet<String> = index
                .weight_map
                .values()
                .map(|s| shard_file_name(s).to_string())
                .collect();
            let mut shards: Vec<String> = names.into_iter().collect();
            shards.sort();
            let map = index
                .weight_map
                .iter()
                .map(|(t, s)| (t.clone(), shard_file_name(s).to_string()))
                .collect();
            (shards, Some(map))
        } else {
            (vec!["model.safetensors".to_string()], None)
        };

    let mut embed_tokens: Option<Vec<BF16>> = None;
    let mut final_norm: Option<Vec<BF16>> = None;
    let mut lm_head_raw: Option<Vec<BF16>> = None;
    let mut layers: Vec<PartialLayer> = (0..config.num_layers)
        .map(|_| PartialLayer::default())
        .collect();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let started = Instant::now();

    for (shard_index, shard) in shards.iter().enumerate() {
        let shard_started = Instant::now();
        let file = std::fs::File::open(model_dir.join(shard))
            .map_err(|e| HerbertError::ModelLoad(format!("cannot open shard {shard}: {e}")))?;
        // SAFETY: the file is read-only and not modified while mapped.
        let mapped = unsafe { Mmap::map(&file) }
            .map_err(|e| HerbertError::ModelLoad(format!("cannot map shard {shard}: {e}")))?;
        let tensors = SafeTensors::deserialize(&mapped[..])
            .map_err(|e| HerbertError::ModelLoad(format!("cannot parse shard {shard}: {e}")))?;
        let names: Vec<String> = match &tensor_shard {
            Some(map) => map
                .iter()
                .filter(|(_, s)| *s == shard)
                .map(|(t, _)| t.clone())
                .collect(),
            None => tensors.names().into_iter().map(|s| s.to_string()).collect(),
        };

        // Packing is CPU-bound: the shard's tensors are split across threads.
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(names.len().max(1));
        let chunk = names.len().div_ceil(workers);
        let read_one = |name: &str| -> Result<Loaded> {
            let view = tensors.tensor(name).map_err(|e| {
                HerbertError::ModelLoad(format!("tensor {name} missing from {shard}: {e}"))
            })?;
            Ok(match classify(name) {
                Kind::Embedding => Loaded::Embedding(load_bf16_from_view(&view, name)?),
                Kind::FinalNorm => Loaded::FinalNorm(load_bf16_from_view(&view, name)?),
                Kind::LmHead => Loaded::LmHead(load_bf16_from_view(&view, name)?),
                Kind::Bf16 { layer, field } => Loaded::Bf16 {
                    layer,
                    field,
                    data: load_bf16_from_view(&view, name)?,
                },
                Kind::F32 { layer, field } => Loaded::F32 {
                    layer,
                    field,
                    data: load_f32_from_view(&view, name)?,
                },
                Kind::Matrix { layer, field } => {
                    let shape = view.shape();
                    if shape.len() != 2 {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: a matrix has 2 dimensions, not {shape:?}"
                        )));
                    }
                    let mut data = load_bf16_from_view(&view, name)?;
                    if let (Some(g), "o_proj" | "down_proj" | "out_proj") = (&config.granite, field)
                    {
                        fold(&mut data, g.residual_multiplier, false);
                    }
                    // Granite 4's shared MLP beside experts is added ungated; the expert block
                    // gates its shared expert by σ(shared_gate · x), here σ(0)
                    // = ½ exactly: its `down` is doubled, exactly.
                    if field == "down_proj" && config.mamba_or_hybrid_moe() {
                        fold(&mut data, 2.0, false);
                    }
                    Loaded::Matrix {
                        layer,
                        field,
                        weight: pack(&data, shape[0], shape[1], name)?,
                    }
                }
                Kind::MambaInProj { layer } => {
                    let shape = view.shape();
                    let (key, value, nv) = (
                        config.linear_num_key_heads * config.linear_key_head_dim,
                        config.linear_num_value_heads * config.linear_value_head_dim,
                        config.linear_num_value_heads,
                    );
                    let conv = 2 * key + value;
                    if shape.len() != 2
                        || shape[0] != value + conv + nv
                        || shape[1] != config.hidden_size
                    {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: expected [{}, {}], got {shape:?}",
                            value + conv + nv,
                            config.hidden_size
                        )));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    let h = shape[1];
                    let z = &data[..value * h];
                    let xbc: Vec<BF16> = (0..conv)
                        .flat_map(|c| {
                            let o = value + mamba_channel(c, key, value);
                            data[o * h..(o + 1) * h].iter().copied()
                        })
                        .collect();
                    let dt: Vec<f32> = data[(value + conv) * h..]
                        .iter()
                        .map(|&w| bf16_to_f32(w))
                        .collect();
                    Loaded::MambaInProj {
                        layer,
                        z: pack(z, value, h, name)?,
                        xbc: pack(&xbc, conv, h, name)?,
                        dt,
                    }
                }
                Kind::ShortConvInProj { layer } => {
                    let shape = view.shape();
                    let h = config.hidden_size;
                    if shape.len() != 2 || shape[0] != 3 * h || shape[1] != h {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: expected [{}, {h}], got {shape:?}",
                            3 * h
                        )));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    let part = |i: usize| pack(&data[i * h * h..(i + 1) * h * h], h, h, name);
                    Loaded::ShortConvInProj {
                        layer,
                        b: part(0)?,
                        c: part(1)?,
                        x: part(2)?,
                    }
                }
                Kind::Expert {
                    layer,
                    expert,
                    field,
                } => {
                    let shape = view.shape();
                    if shape.len() != 2 {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: a matrix has 2 dimensions, not {shape:?}"
                        )));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    Loaded::Expert {
                        layer,
                        expert,
                        field,
                        weight: pack(&data, shape[0], shape[1], name)?,
                    }
                }
                Kind::SharedGateUp { layer } => {
                    let shape = view.shape();
                    if shape.len() != 2 || shape[0] % 2 != 0 {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: expected [2 · inter, hidden], got {shape:?}"
                        )));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    let (inter, h) = (shape[0] / 2, shape[1]);
                    Loaded::SharedGateUp {
                        layer,
                        gate: pack(&data[..inter * h], inter, h, name)?,
                        up: pack(&data[inter * h..], inter, h, name)?,
                    }
                }
                Kind::Router { layer } => {
                    let shape = view.shape();
                    if shape.len() != 2 || shape[0] % 16 != 0 {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: expected [experts, hidden], got {shape:?}"
                        )));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    // Sixteen experts a block: a matrix of fewer than 64 rows stays bf16 whatever
                    // `--q4` says.
                    let blocks = (0..shape[0] / 16)
                        .map(|b| {
                            pack(
                                &data[b * 16 * shape[1]..(b + 1) * 16 * shape[1]],
                                16,
                                shape[1],
                                name,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Loaded::Router { layer, blocks }
                }
                Kind::Experts { layer, field } => {
                    let shape = view.shape();
                    if shape.len() != 3 {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: expected [experts, rows, columns], got {shape:?}"
                        )));
                    }
                    let mut data = load_bf16_from_view(&view, name)?;
                    if let (Some(g), "down") = (&config.granite, field) {
                        fold(&mut data, g.residual_multiplier, false);
                    }
                    if shape[0] == 0 {
                        return Err(HerbertError::ModelLoad(format!("{name}: no expert")));
                    }
                    let per = shape[1] * shape[2];
                    // Qwen3-VL keeps its experts transposed, `gate_up_proj` `[experts, hidden,
                    // 2·inter]` and `down_proj` `[experts, inter, hidden]`
                    // (`torch.bmm(x, w)`); transformers 5 transposes them at
                    // load (`Transpose(1, 2)`), and so does this loader, one expert at a time.
                    let transposed = config.qwen_vl.is_some();
                    let (rows, cols) = if transposed {
                        (shape[2], shape[1])
                    } else {
                        (shape[1], shape[2])
                    };
                    let transpose = |e: &[BF16]| -> Vec<BF16> {
                        let mut t = Vec::with_capacity(e.len());
                        for r in 0..rows {
                            t.extend((0..cols).map(|c| e[c * rows + r]));
                        }
                        t
                    };
                    let transpose = &transpose;
                    // A gigabyte a tensor: its experts packed on every core, slices of experts
                    // each.
                    let threads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                        .min(shape[0])
                        .max(1);
                    let slice = shape[0].div_ceil(threads);
                    let (data, pack) = (&data, &pack);
                    let weights =
                        std::thread::scope(|scope| -> Result<Vec<PackedBf16R16Weight>> {
                            let handles: Vec<_> = (0..shape[0])
                                .step_by(slice)
                                .map(|first| {
                                    scope.spawn(move || {
                                        (first..(first + slice).min(shape[0]))
                                            .map(|e| {
                                                let w = &data[e * per..(e + 1) * per];
                                                if transposed {
                                                    pack(&transpose(w), rows, cols, name)
                                                } else {
                                                    pack(w, rows, cols, name)
                                                }
                                            })
                                            .collect::<Result<Vec<_>>>()
                                    })
                                })
                                .collect();
                            let mut all = Vec::with_capacity(shape[0]);
                            for h in handles {
                                all.extend(h.join().map_err(|_| {
                                    HerbertError::ModelLoad(
                                        "an expert packing thread panicked".into(),
                                    )
                                })??);
                            }
                            Ok(all)
                        })?;
                    Loaded::Experts {
                        layer,
                        field,
                        weights,
                    }
                }
                Kind::Skipped => {
                    Loaded::Skipped(name.split('.').take(2).collect::<Vec<_>>().join("."))
                }
            })
        };
        let loaded: Vec<Loaded> =
            std::thread::scope(|scope| -> Result<Vec<Loaded>> {
                let handles: Vec<_> = names
                    .chunks(chunk.max(1))
                    .map(|slice| {
                        scope.spawn(move || {
                            slice
                                .iter()
                                .map(|n| read_one(n))
                                .collect::<Result<Vec<_>>>()
                        })
                    })
                    .collect();
                let mut all = Vec::with_capacity(names.len());
                for handle in handles {
                    all.extend(handle.join().map_err(|_| {
                        HerbertError::ModelLoad("a loader thread panicked".into())
                    })??);
                }
                Ok(all)
            })?;

        for item in loaded {
            match item {
                Loaded::Embedding(v) => embed_tokens = Some(v),
                Loaded::FinalNorm(v) => final_norm = Some(v),
                Loaded::LmHead(v) => lm_head_raw = Some(v),
                Loaded::Bf16 { layer, field, data } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .bf16
                        .insert(field, data);
                }
                Loaded::F32 { layer, field, data } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .f32
                        .insert(field, data);
                }
                Loaded::Matrix {
                    layer,
                    field,
                    weight,
                } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .matrices
                        .insert(field, weight);
                }
                Loaded::Router { layer, blocks } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .router = blocks;
                }
                Loaded::Experts {
                    layer,
                    field,
                    weights,
                } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .experts
                        .insert(field, weights);
                }
                Loaded::MambaInProj { layer, z, xbc, dt } => {
                    let l = layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?;
                    l.matrices.insert("in_proj_z", z);
                    l.matrices.insert("in_proj_qkv", xbc);
                    l.f32.insert("in_proj_a", dt);
                }
                Loaded::ShortConvInProj { layer, b, c, x } => {
                    let l = layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?;
                    l.matrices.insert("in_proj_z", b);
                    l.matrices.insert("in_proj_c", c);
                    l.matrices.insert("in_proj_qkv", x);
                }
                Loaded::Expert {
                    layer,
                    expert,
                    field,
                    weight,
                } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .expert
                        .insert((field, expert), weight);
                }
                Loaded::SharedGateUp { layer, gate, up } => {
                    let l = layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?;
                    l.matrices.insert("gate_proj", gate);
                    l.matrices.insert("up_proj", up);
                }
                Loaded::Skipped(prefix) => *skipped.entry(prefix).or_default() += 1,
            }
        }
        if show_progress {
            eprintln!(
                "[loader] shard {}/{}: {} tensors, {:.2} GiB mapped, {:.1} s",
                shard_index + 1,
                shards.len(),
                names.len(),
                mapped.len() as f64 / (1u64 << 30) as f64,
                shard_started.elapsed().as_secs_f64()
            );
        }
    }
    if !skipped.is_empty() {
        let list: Vec<String> = skipped.iter().map(|(p, n)| format!("{p}.* ×{n}")).collect();
        eprintln!(
            "[loader] skipped, not used by the engine: {}",
            list.join(", ")
        );
    }

    let mut embed_tokens = embed_tokens
        .ok_or_else(|| HerbertError::ModelLoad("missing model.embed_tokens.weight".into()))?;
    let norm =
        final_norm.ok_or_else(|| HerbertError::ModelLoad("missing model.norm.weight".into()))?;
    // The language head: its own matrix, or the embedding table when tied. Granite 3 divides the
    // logits by `logits_scaling` (folded into the head) and multiplies the embeddings by
    // `embedding_multiplier` (folded into the table, after the tied head took its copy).
    let head = |raw: &[BF16], name: &str| -> Result<PackedBf16R16Weight> {
        match &config.granite {
            Some(g) => {
                let mut scaled = raw.to_vec();
                fold(&mut scaled, g.logits_scaling, true);
                pack(&scaled, config.vocab_size, config.hidden_size, name)
            }
            None => pack(raw, config.vocab_size, config.hidden_size, name),
        }
    };
    let lm_head = match lm_head_raw {
        Some(raw) => head(&raw, "lm_head.weight")?,
        None if config.tie_word_embeddings => head(&embed_tokens, "lm_head.weight (tied)")?,
        None => {
            return Err(HerbertError::ModelLoad(
                "missing lm_head.weight and tie_word_embeddings is false".into(),
            ))
        }
    };
    if let Some(g) = &config.granite {
        fold(&mut embed_tokens, g.embedding_multiplier, false);
    }
    let decoder_layers: Vec<DecoderLayer> = layers
        .into_iter()
        .enumerate()
        .map(|(l, partial)| partial.finalize(l, &config))
        .collect::<Result<_>>()?;
    let (cos_cache, sin_cache) = compute_rope_cache(&config);
    if show_progress {
        eprintln!(
            "[loader] {} layers assembled, RoPE tables built, {:.1} s in all",
            decoder_layers.len(),
            started.elapsed().as_secs_f64()
        );
    }
    let model = Model {
        embed_tokens: AVec::from_vec(embed_tokens),
        decoder_layers,
        norm: AVec::from_vec(norm),
        lm_head,
        config: config.clone(),
        cos_cache: AVec::from_vec(cos_cache),
        sin_cache: AVec::from_vec(sin_cache),
        gemma: None,
        rope_sliding: config
            .lfm3
            .as_ref()
            .filter(|c| c.sliding.contains(&true))
            .map(|c| {
                let (cos, sin) = compute_rope_sliding(&config, c);
                (AVec::from_vec(cos), AVec::from_vec(sin))
            }),
        rope_nope: config.no_rope.contains(&true).then(|| {
            let n = config.max_position_embeddings * crate::model::model::NOPE_ROTARY_NDIMS / 2;
            (
                AVec::from_vec(vec![1.0f32; n]),
                AVec::from_vec(vec![0.0f32; n]),
            )
        }),
    };
    Ok((config, model))
}

/// The file name of a `weight_map` entry: whatever folders it names, the shard is read from the
/// model's own folder.
pub(crate) fn shard_file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn out_of_range(layer: usize, config: &Config) -> HerbertError {
    HerbertError::ModelLoad(format!(
        "tensor of layer {layer} but the config declares {} layers",
        config.num_layers
    ))
}

#[cfg(test)]
mod qwen_vl_tests {
    use super::*;

    /// The text model of Qwen3-VL-30B-A3B-Instruct, its `config.json` cut to what the reader takes.
    const VL: &str = r#"{"model_type": "qwen3_vl_moe", "tie_word_embeddings": false,
        "text_config": {"attention_bias": false, "decoder_sparse_step": 1, "head_dim": 128, "hidden_act": "silu",
        "hidden_size": 2048, "intermediate_size": 6144, "max_position_embeddings": 4096, "mlp_only_layers": [],
        "model_type": "qwen3_vl_moe_text", "moe_intermediate_size": 768, "norm_topk_prob": true,
        "num_attention_heads": 32, "num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 48,
        "num_key_value_heads": 4, "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000, "vocab_size": 151936},
        "vision_config": {"deepstack_visual_indexes": [8, 16, 24]}}"#;

    #[test]
    fn qwen3_vl_config() {
        let c = Config::from_json(VL).unwrap();
        let m = c.qwen_moe.as_ref().unwrap();
        assert!(
            c.qk_norm
                && !c.qkv_bias
                && c.intermediate_size == 0
                && c.rotary_ndims == 128
                && c.rope_theta == 5e6
        );
        assert!(
            m.shared_inter == 0
                && m.norm_topk_prob
                && m.num_experts == 128
                && m.top_k == 8
                && m.inter == 768
                && m.sparse.iter().all(|&s| s)
        );
        assert_eq!(
            c.qwen_vl,
            Some(QwenVlConfig {
                mrope_section: [24, 20, 20],
                interleaved: true,
                deepstack_layers: 3
            })
        );
        assert_eq!(c.mlp_width(0), 0);
    }

    /// The axis of each frequency as `Qwen3VLMoeTextRotaryEmbedding.recomposition_frequencies`
    /// (transformers 5.22.1) lays it out (`freq[a] = a`, the first half of its output), and the
    /// row of `(t, h, w) = (5, 17, 29)`: its first eight cosines in f32 from the same module.
    #[test]
    fn mrope_matches_transformers() {
        let c = Config::from_json(VL).unwrap();
        let vl = c.qwen_vl.as_ref().unwrap();
        let mut expected: Vec<usize> = (0..60).map(|i| i % 3).collect();
        expected.extend([0; 4]);
        assert_eq!(
            (0..64).map(|i| vl.mrope_axis(i)).collect::<Vec<_>>(),
            expected
        );
        let inv = rope_inv_freq(&c);
        let (cos, _) = mrope_row(vl, &inv, [5, 17, 29]);
        let torch = [
            0.283_662_2,
            0.701_896_85,
            0.588_799_83,
            -0.754_942_66,
            0.980_141_04,
            -0.742_263_8,
            0.383_283_6,
            -0.999_990_6f32,
        ];
        for (a, b) in cos.iter().zip(torch) {
            assert!((a - b).abs() <= 2e-6, "{a} vs {b}");
        }
    }

    /// A text token (`t = h = w`) gets the plain RoPE's row, bit for bit; three different positions
    /// take each frequency from its own axis's row.
    #[test]
    fn mrope_text_rows_are_plain_rope() {
        let c = Config::from_json(VL).unwrap();
        let vl = c.qwen_vl.clone().unwrap();
        let (cos, sin) = compute_rope_cache(&c);
        let plain = Config {
            qwen_vl: None,
            ..c.clone()
        };
        let (pc, ps) = compute_rope_cache(&plain);
        assert!(
            cos.iter().zip(&pc).all(|(a, b)| a.to_bits() == b.to_bits())
                && sin.iter().zip(&ps).all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let inv = rope_inv_freq(&c);
        let p = [7usize, 300, 4001];
        let (rc, rs) = mrope_row(&vl, &inv, p);
        for i in 0..64 {
            let q = p[vl.mrope_axis(i)];
            assert_eq!(rc[i].to_bits(), pc[q * 64 + i].to_bits());
            assert_eq!(rs[i].to_bits(), ps[q * 64 + i].to_bits());
        }
    }
}
