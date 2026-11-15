//! A Gemma 4 text checkpoint (E2B, E4B, 12B, 31B, and the 26B-A4B mixture of experts) read into
//! the model structures.
//!
//! The tensors live under `model.language_model.`; the vision and audio towers are skipped. Per
//! layer: four norms around the attention and the MLP (`input_layernorm`,
//! `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`), the
//! per-head `q_norm` / `k_norm` (the layers that read another layer's cache have no `k_proj`,
//! `v_proj` or `k_norm`), the per-layer input branch (`per_layer_input_gate`,
//! `per_layer_projection`, `post_per_layer_input_norm`) and a `layer_scalar`. At the model level:
//! the embedding table (also the language head), the table of per-layer embeddings
//! (`embed_tokens_per_layer`, `[vocab, layers · pl_dim]`: 4.7 GB on the E2B, mapped and never
//! copied: a token reads one row), the projection of the embedding into the per-layer inputs and
//! its norm.

use crate::core::avec::AVec;
use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::core::tensor::BF16;
use crate::model::loader::{load_bf16_from_view, resolve_model_dir};
use crate::model::model::{Attention, DecoderLayer, LayerBlock, Mlp, Model};
use crate::model::weight::PackedBf16R16Weight;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Positions the RoPE tables cover at most (the model declares 131 072).
pub const MAX_TABLE_POSITIONS: usize = 32768;

/// A bf16 table that stays in the mapping of the checkpoint.
pub struct Mapped {
    _map: Arc<Mmap>,
    ptr: *const u16,
    len: usize,
}

// SAFETY: read-only view of a read-only mapping kept alive by `_map`.
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl std::ops::Deref for Mapped {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        // SAFETY: `ptr..ptr+len` lies in the mapping `_map` holds, aligned for u16 (checked at
        // construction).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// What a Gemma layer has beyond the dense layer.
pub struct GemmaLayer {
    /// `post_attention_layernorm`: applied to the attention output before it joins the residual.
    pub post_attn_out_norm: AVec<BF16>,
    /// `pre_feedforward_layernorm`: the norm of the MLP input.
    pub pre_ff_norm: AVec<BF16>,
    /// `post_feedforward_layernorm`: applied to the MLP output before it joins the residual.
    pub post_ff_norm: AVec<BF16>,
    /// Sliding window (positions seen), 0 = full attention.
    pub window: usize,
    /// The layer whose cache this layer reads, `None` when it owns one.
    pub kv_source: Option<usize>,
    /// The per-layer input branch; `None` on a model without per-layer inputs (the 12B, the 26B,
    /// the 31B).
    pub ple: Option<PleLayer>,
    /// The layer has no `v_proj`: its values are its keys, before the key norm and the RoPE
    /// (`attention_k_eq_v`).
    pub k_eq_v: bool,
    /// The mixture-of-experts block (the 26B-A4B).
    pub moe: Option<MoeLayer>,
    pub layer_scalar: f32,
}

/// The mixture-of-experts block of a layer, which runs in parallel to the dense MLP: the router
/// chooses `top_k` of the experts from the residual, the experts read the residual normed by
/// `pre_ff_norm_2`.
pub struct MoeLayer {
    /// `router.proj`, `[experts, hidden]`, kept exact in bf16 as sixteen-row matrices (a selection
    /// must not move with a quantization): block `b` holds the experts `16b..16b+15`.
    pub router: Vec<PackedBf16R16Weight>,
    /// `router.scale · hidden^-½`, the weight of the norm in front of the router.
    pub router_scale: AVec<BF16>,
    /// `router.per_expert_scale`, f32.
    pub per_expert_scale: AVec<f32>,
    /// `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1` (dense branch),
    /// `post_feedforward_layernorm_2`.
    pub pre_ff_norm_2: AVec<BF16>,
    pub post_ff_norm_1: AVec<BF16>,
    pub post_ff_norm_2: AVec<BF16>,
    /// Per expert: `[2 · inter, hidden]` (gate rows then up rows) and `[hidden, inter]`.
    pub gate_up: Vec<PackedBf16R16Weight>,
    pub down: Vec<PackedBf16R16Weight>,
}

/// The weights of the per-layer input branch of a layer.
pub struct PleLayer {
    /// `[pl_dim, hidden]`.
    pub gate: PackedBf16R16Weight,
    /// `[hidden, pl_dim]`.
    pub proj: PackedBf16R16Weight,
    /// `[hidden]`.
    pub norm: AVec<BF16>,
}

/// The per-layer inputs of a model that has them.
pub struct PleModel {
    /// `embed_tokens_per_layer`, `[vocab_size_per_layer_input, layers · pl_dim]`.
    pub table: Mapped,
    /// `per_layer_model_projection`, `[layers · pl_dim, hidden]`.
    pub model_proj: PackedBf16R16Weight,
    /// `per_layer_projection_norm`, `[pl_dim]`.
    pub proj_norm: AVec<BF16>,
}

/// What a Gemma model has beyond the dense model.
pub struct GemmaModel {
    /// The per-layer inputs; `None` when `hidden_size_per_layer_input` is 0.
    pub ple: Option<PleModel>,
    /// Tables of the full-attention layers: `(rotary_ndims, cos, sin)`, f32, `[positions,
    /// rotary_ndims/2]`.
    pub rope_full: (usize, AVec<f32>, AVec<f32>),
}

/// `cos`/`sin` of `pos · inv_freq[i]`.
fn table(inv_freq: &[f32], positions: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0.0f32; positions * half];
    let mut sin = vec![0.0f32; positions * half];
    for pos in 0..positions {
        for (i, f) in inv_freq.iter().enumerate() {
            let angle = pos as f32 * f;
            cos[pos * half + i] = angle.cos();
            sin[pos * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// The sliding layers rotate every dimension of the head (`rotate_half` pairs `(i, i + hd/2)`); the
/// full layers use the proportional RoPE of the reference: the first `partial · hd / 2` frequencies
/// are `base^(−2i/hd)` (exponent over the **whole** head), the others are zero, so those pairs do
/// not rotate.
fn rope_tables(config: &Config) -> ((Vec<f32>, Vec<f32>), (usize, Vec<f32>, Vec<f32>)) {
    let g = config.gemma.as_ref().expect("gemma config");
    let positions = config.max_position_embeddings.min(MAX_TABLE_POSITIONS);
    let hd = config.head_dim;
    let sliding: Vec<f32> = (0..hd / 2)
        .map(|i| 1.0 / g.rope_theta_sliding.powf((2 * i) as f32 / hd as f32))
        .collect();
    let ghd = g.global_head_dim;
    let rotated = (ghd as f32 * g.full_partial_rotary_factor) as usize / 2;
    let full: Vec<f32> = (0..ghd / 2)
        .map(|i| {
            if i < rotated {
                1.0 / g.rope_theta_full.powf((2 * i) as f32 / ghd as f32)
            } else {
                0.0
            }
        })
        .collect();
    let (c, s) = table(&full, positions);
    (table(&sliding, positions), (ghd, c, s))
}

enum Kind {
    Embedding,
    PerLayerTable,
    FinalNorm,
    ModelProjection,
    ModelProjectionNorm,
    Norm { layer: usize, field: &'static str },
    Scalar { layer: usize },
    Matrix { layer: usize, field: &'static str },
    Router { layer: usize },
    Experts { layer: usize, field: &'static str },
    Skipped,
}

fn classify(name: &str) -> Kind {
    let Some(rest) = name.strip_prefix("model.language_model.") else {
        return Kind::Skipped;
    };
    match rest {
        "embed_tokens.weight" => return Kind::Embedding,
        "embed_tokens_per_layer.weight" => return Kind::PerLayerTable,
        "norm.weight" => return Kind::FinalNorm,
        "per_layer_model_projection.weight" => return Kind::ModelProjection,
        "per_layer_projection_norm.weight" => return Kind::ModelProjectionNorm,
        _ => {}
    }
    let Some(rest) = rest.strip_prefix("layers.") else {
        return Kind::Skipped;
    };
    let Some((index, field)) = rest.split_once('.') else {
        return Kind::Skipped;
    };
    let Ok(layer) = index.parse::<usize>() else {
        return Kind::Skipped;
    };
    match field {
        "input_layernorm.weight" => Kind::Norm {
            layer,
            field: "input_layernorm",
        },
        "post_attention_layernorm.weight" => Kind::Norm {
            layer,
            field: "post_attention_layernorm",
        },
        "pre_feedforward_layernorm.weight" => Kind::Norm {
            layer,
            field: "pre_feedforward_layernorm",
        },
        "post_feedforward_layernorm.weight" => Kind::Norm {
            layer,
            field: "post_feedforward_layernorm",
        },
        "post_per_layer_input_norm.weight" => Kind::Norm {
            layer,
            field: "post_per_layer_input_norm",
        },
        "self_attn.q_norm.weight" => Kind::Norm {
            layer,
            field: "q_norm",
        },
        "self_attn.k_norm.weight" => Kind::Norm {
            layer,
            field: "k_norm",
        },
        "layer_scalar" => Kind::Scalar { layer },
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
        "pre_feedforward_layernorm_2.weight" => Kind::Norm {
            layer,
            field: "pre_feedforward_layernorm_2",
        },
        "post_feedforward_layernorm_1.weight" => Kind::Norm {
            layer,
            field: "post_feedforward_layernorm_1",
        },
        "post_feedforward_layernorm_2.weight" => Kind::Norm {
            layer,
            field: "post_feedforward_layernorm_2",
        },
        "router.scale" => Kind::Norm {
            layer,
            field: "router_scale",
        },
        "router.per_expert_scale" => Kind::Norm {
            layer,
            field: "per_expert_scale",
        },
        "router.proj.weight" => Kind::Router { layer },
        "experts.gate_up_proj" => Kind::Experts {
            layer,
            field: "gate_up",
        },
        "experts.down_proj" => Kind::Experts {
            layer,
            field: "down",
        },
        "per_layer_input_gate.weight" => Kind::Matrix {
            layer,
            field: "per_layer_input_gate",
        },
        "per_layer_projection.weight" => Kind::Matrix {
            layer,
            field: "per_layer_projection",
        },
        _ => Kind::Skipped,
    }
}

enum Loaded {
    Embedding(Vec<BF16>),
    PerLayerTable(Mapped),
    FinalNorm(Vec<BF16>),
    ModelProjection(PackedBf16R16Weight),
    ModelProjectionNorm(Vec<BF16>),
    Norm {
        layer: usize,
        field: &'static str,
        data: Vec<BF16>,
    },
    Scalar {
        layer: usize,
        value: f32,
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
    Skipped,
}

#[derive(Default)]
struct Partial {
    norms: HashMap<&'static str, Vec<BF16>>,
    matrices: HashMap<&'static str, PackedBf16R16Weight>,
    scalar: Option<f32>,
    router: Vec<PackedBf16R16Weight>,
    experts: HashMap<&'static str, Vec<PackedBf16R16Weight>>,
}

/// Read the checkpoint at `model_dir`. `pack(matrix, rows, columns, name)` turns a row-major bf16
/// matrix into the engine's weight; it runs on several threads.
pub fn load_model(
    model_dir: &Path,
    pack: impl Fn(&[BF16], usize, usize, &str) -> Result<PackedBf16R16Weight> + Send + Sync,
    show_progress: bool,
) -> Result<(Config, Model)> {
    let model_dir = resolve_model_dir(model_dir)?;
    let config = Config::from_file(&model_dir.join("config.json"))?;
    let g = config
        .gemma
        .clone()
        .ok_or_else(|| HerbertError::ModelLoad("not a Gemma 4 config".into()))?;
    let started = Instant::now();
    let shards: Vec<String> = {
        let index = model_dir.join("model.safetensors.index.json");
        if index.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let idx: Index = serde_json::from_slice(&std::fs::read(&index)?)?;
            let mut v: Vec<String> = idx
                .weight_map
                .values()
                .map(|s| crate::model::loader::shard_file_name(s).to_string())
                .collect();
            v.sort();
            v.dedup();
            v
        } else {
            vec!["model.safetensors".to_string()]
        }
    };
    let n = config.num_layers;
    let mut embed: Option<Vec<BF16>> = None;
    let mut ple_table: Option<Mapped> = None;
    let mut final_norm: Option<Vec<BF16>> = None;
    let mut model_proj: Option<PackedBf16R16Weight> = None;
    let mut model_proj_norm: Option<Vec<BF16>> = None;
    let mut layers: Vec<Partial> = (0..n).map(|_| Partial::default()).collect();
    for shard in &shards {
        let file = std::fs::File::open(model_dir.join(shard))
            .map_err(|e| HerbertError::ModelLoad(format!("cannot open shard {shard}: {e}")))?;
        // SAFETY: the file is read-only and not modified while mapped.
        let mapped = Arc::new(
            unsafe { Mmap::map(&file) }
                .map_err(|e| HerbertError::ModelLoad(format!("cannot map shard {shard}: {e}")))?,
        );
        let tensors = SafeTensors::deserialize(&mapped[..])
            .map_err(|e| HerbertError::ModelLoad(format!("cannot parse shard {shard}: {e}")))?;
        let names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(names.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let read_one = |name: &str| -> Result<Loaded> {
            let view = tensors.tensor(name).map_err(|e| {
                HerbertError::ModelLoad(format!("tensor {name} missing from {shard}: {e}"))
            })?;
            Ok(match classify(name) {
                Kind::Embedding => Loaded::Embedding(load_bf16_from_view(&view, name)?),
                Kind::PerLayerTable => {
                    let data = view.data();
                    if view.dtype() != safetensors::Dtype::BF16
                        || !(data.as_ptr() as usize).is_multiple_of(2)
                    {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: expected an aligned bf16 table"
                        )));
                    }
                    Loaded::PerLayerTable(Mapped {
                        _map: mapped.clone(),
                        ptr: data.as_ptr().cast(),
                        len: data.len() / 2,
                    })
                }
                Kind::FinalNorm => Loaded::FinalNorm(load_bf16_from_view(&view, name)?),
                Kind::ModelProjectionNorm => {
                    Loaded::ModelProjectionNorm(load_bf16_from_view(&view, name)?)
                }
                Kind::ModelProjection | Kind::Matrix { .. } => {
                    let shape = view.shape();
                    if shape.len() != 2 {
                        return Err(HerbertError::ModelLoad(format!(
                            "{name}: a matrix has 2 dimensions, not {shape:?}"
                        )));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    let weight = pack(&data, shape[0], shape[1], name)?;
                    match classify(name) {
                        Kind::Matrix { layer, field } => Loaded::Matrix {
                            layer,
                            field,
                            weight,
                        },
                        _ => Loaded::ModelProjection(weight),
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
                    // `--int8` says.
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
                    let data = load_bf16_from_view(&view, name)?;
                    let per = shape[1] * shape[2];
                    let weights = (0..shape[0])
                        .map(|e| pack(&data[e * per..(e + 1) * per], shape[1], shape[2], name))
                        .collect::<Result<Vec<_>>>()?;
                    Loaded::Experts {
                        layer,
                        field,
                        weights,
                    }
                }
                Kind::Norm { layer, field } => Loaded::Norm {
                    layer,
                    field,
                    data: load_bf16_from_view(&view, name)?,
                },
                Kind::Scalar { layer } => {
                    let v = load_bf16_from_view(&view, name)?;
                    let first = *v
                        .first()
                        .ok_or_else(|| HerbertError::ModelLoad(format!("{name}: empty")))?;
                    Loaded::Scalar {
                        layer,
                        value: crate::core::tensor::bf16_to_f32(first),
                    }
                }
                Kind::Skipped => Loaded::Skipped,
            })
        };
        let loaded: Vec<Loaded> =
            std::thread::scope(|scope| -> Result<Vec<Loaded>> {
                let handles: Vec<_> = (0..workers)
                    .map(|_| {
                        scope.spawn(|| {
                            let mut out = Vec::new();
                            loop {
                                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let Some(name) = names.get(i) else { break };
                                out.push(read_one(name));
                            }
                            out.into_iter().collect::<Result<Vec<_>>>()
                        })
                    })
                    .collect();
                let mut all = Vec::with_capacity(names.len());
                for h in handles {
                    all.extend(h.join().map_err(|_| {
                        HerbertError::ModelLoad("a loader thread panicked".into())
                    })??);
                }
                Ok(all)
            })?;
        for item in loaded {
            match item {
                Loaded::Embedding(v) => embed = Some(v),
                Loaded::PerLayerTable(m) => ple_table = Some(m),
                Loaded::FinalNorm(v) => final_norm = Some(v),
                Loaded::ModelProjection(w) => model_proj = Some(w),
                Loaded::ModelProjectionNorm(v) => model_proj_norm = Some(v),
                Loaded::Norm { layer, field, data } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .norms
                        .insert(field, data);
                }
                Loaded::Scalar { layer, value } => {
                    layers
                        .get_mut(layer)
                        .ok_or_else(|| out_of_range(layer, &config))?
                        .scalar = Some(value);
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
                Loaded::Skipped => {}
            }
        }
        if show_progress {
            eprintln!(
                "[loader] shard {shard}: {} tensors, {:.2} GiB mapped, {:.1} s",
                names.len(),
                mapped.len() as f64 / (1u64 << 30) as f64,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let missing = |what: &str| HerbertError::ModelLoad(format!("missing {what}"));
    let embed = embed.ok_or_else(|| missing("model.language_model.embed_tokens.weight"))?;
    let lm_head = pack(
        &embed,
        config.vocab_size,
        config.hidden_size,
        "lm_head.weight (tied embedding)",
    )?;
    let has_ple = g.pl_dim > 0;
    let ple = if has_ple {
        let table = ple_table
            .ok_or_else(|| missing("model.language_model.embed_tokens_per_layer.weight"))?;
        // The generated lookup reads the row of the token's own id (`lowering.rs`): a smaller
        // table, whose missing ids the reference maps to row 0, is refused instead of read past.
        if g.vocab_size_per_layer_input < config.vocab_size {
            return Err(HerbertError::ModelLoad(format!(
                "vocab_size_per_layer_input {} below vocab_size {}: not served",
                g.vocab_size_per_layer_input, config.vocab_size
            )));
        }
        if table.len != g.vocab_size_per_layer_input * n * g.pl_dim {
            return Err(HerbertError::ModelLoad(format!(
                "embed_tokens_per_layer holds {} values, expected {} · {} · {}",
                table.len, g.vocab_size_per_layer_input, n, g.pl_dim
            )));
        }
        Some(PleModel {
            table,
            model_proj: model_proj.ok_or_else(|| missing("per_layer_model_projection.weight"))?,
            proj_norm: AVec::from_vec(
                model_proj_norm.ok_or_else(|| missing("per_layer_projection_norm.weight"))?,
            ),
        })
    } else {
        None
    };
    let mut decoder_layers = Vec::with_capacity(n);
    for (l, mut p) in layers.into_iter().enumerate() {
        let norm = |p: &mut Partial, f: &str| -> Result<AVec<BF16>> {
            p.norms
                .remove(f)
                .map(AVec::from_vec)
                .ok_or_else(|| HerbertError::ModelLoad(format!("layer {l}: missing {f}")))
        };
        let mat = |p: &mut Partial, f: &str| -> Result<PackedBf16R16Weight> {
            p.matrices
                .remove(f)
                .ok_or_else(|| HerbertError::ModelLoad(format!("layer {l}: missing {f}")))
        };
        let sliding = g.layer_sliding[l];
        let head_dim = g.head_dim_of(l, config.head_dim);
        let kv_heads = if sliding {
            config.num_key_value_heads
        } else {
            g.global_kv_heads
        };
        let source = g.kv_source(l);
        let owns = source == l;
        let k_eq_v = g.k_eq_v && !sliding && owns;
        let (k_proj, v_proj, k_norm) = if owns {
            let v = if k_eq_v {
                PackedBf16R16Weight::empty(kv_heads * head_dim, config.hidden_size)
            } else {
                mat(&mut p, "v_proj")?
            };
            (mat(&mut p, "k_proj")?, v, norm(&mut p, "k_norm")?)
        } else {
            (
                PackedBf16R16Weight::empty(kv_heads * head_dim, config.hidden_size),
                PackedBf16R16Weight::empty(kv_heads * head_dim, config.hidden_size),
                AVec::from_vec(Vec::new()),
            )
        };
        let attention = Attention {
            q_proj: mat(&mut p, "q_proj")?,
            k_proj,
            v_proj,
            o_proj: mat(&mut p, "o_proj")?,
            q_norm: norm(&mut p, "q_norm")?,
            k_norm,
            q_bias: AVec::from_vec(Vec::new()),
            k_bias: AVec::from_vec(Vec::new()),
            v_bias: AVec::from_vec(Vec::new()),
            num_heads: config.num_attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            rotary_ndims: head_dim,
        };
        let scalar = p
            .scalar
            .ok_or_else(|| HerbertError::ModelLoad(format!("layer {l}: missing layer_scalar")))?;
        let gemma = GemmaLayer {
            post_attn_out_norm: norm(&mut p, "post_attention_layernorm")?,
            pre_ff_norm: norm(&mut p, "pre_feedforward_layernorm")?,
            post_ff_norm: norm(&mut p, "post_feedforward_layernorm")?,
            window: if sliding { g.sliding_window } else { 0 },
            kv_source: if owns { None } else { Some(source) },
            moe: match g.moe_config() {
                Some(m) => {
                    let router = std::mem::take(&mut p.router);
                    let gate_up = p.experts.remove("gate_up").ok_or_else(|| {
                        HerbertError::ModelLoad(format!("layer {l}: missing experts.gate_up_proj"))
                    })?;
                    let down = p.experts.remove("down").ok_or_else(|| {
                        HerbertError::ModelLoad(format!("layer {l}: missing experts.down_proj"))
                    })?;
                    if router.len() * 16 != m.num_experts
                        || gate_up.len() != m.num_experts
                        || down.len() != m.num_experts
                    {
                        return Err(HerbertError::ModelLoad(format!(
                            "layer {l}: the router or the experts do not count {} experts",
                            m.num_experts
                        )));
                    }
                    // `router.scale · hidden^-½`, rounded to bf16 as the reference multiplies it.
                    let scale = norm(&mut p, "router_scale")?;
                    let root = (config.hidden_size as f32).powf(-0.5);
                    let scaled: Vec<BF16> = scale
                        .iter()
                        .map(|&b| {
                            crate::core::tensor::f32_to_bf16(
                                crate::core::tensor::bf16_to_f32(b) * root,
                            )
                        })
                        .collect();
                    let per = norm(&mut p, "per_expert_scale")?;
                    Some(MoeLayer {
                        router,
                        router_scale: AVec::from_vec(scaled),
                        per_expert_scale: AVec::from_vec(
                            per.iter()
                                .map(|&b| crate::core::tensor::bf16_to_f32(b))
                                .collect(),
                        ),
                        pre_ff_norm_2: norm(&mut p, "pre_feedforward_layernorm_2")?,
                        post_ff_norm_1: norm(&mut p, "post_feedforward_layernorm_1")?,
                        post_ff_norm_2: norm(&mut p, "post_feedforward_layernorm_2")?,
                        gate_up,
                        down,
                    })
                }
                None => None,
            },
            ple: if has_ple {
                Some(PleLayer {
                    gate: mat(&mut p, "per_layer_input_gate")?,
                    proj: mat(&mut p, "per_layer_projection")?,
                    norm: norm(&mut p, "post_per_layer_input_norm")?,
                })
            } else {
                None
            },
            k_eq_v,
            layer_scalar: scalar,
        };
        let gate_proj = mat(&mut p, "gate_proj")?;
        let intermediate_size = g.intermediate_of(l, config.intermediate_size);
        if gate_proj.n != intermediate_size {
            return Err(HerbertError::ModelLoad(format!(
                "layer {l}: gate_proj has {} rows, expected {intermediate_size}",
                gate_proj.n
            )));
        }
        decoder_layers.push(DecoderLayer {
            input_layernorm: norm(&mut p, "input_layernorm")?,
            block: LayerBlock::Attention(attention),
            // The role: the norm in front of the MLP.
            post_attention_layernorm: gemma.pre_ff_norm.clone(),
            mlp: Mlp {
                gate_proj,
                up_proj: mat(&mut p, "up_proj")?,
                down_proj: mat(&mut p, "down_proj")?,
                intermediate_size,
            },
            gemma: Some(gemma),
            qwen_moe: None,
            lfm3: None,
        });
    }
    let ((cos, sin), (full_ndims, fcos, fsin)) = rope_tables(&config);
    let gemma_model = GemmaModel {
        ple,
        rope_full: (full_ndims, AVec::from_vec(fcos), AVec::from_vec(fsin)),
    };
    if show_progress {
        eprintln!(
            "[loader] {} layers assembled, RoPE tables built, {:.1} s in all",
            decoder_layers.len(),
            started.elapsed().as_secs_f64()
        );
    }
    let model = Model {
        embed_tokens: AVec::from_vec(embed),
        decoder_layers,
        norm: AVec::from_vec(
            final_norm.ok_or_else(|| missing("model.language_model.norm.weight"))?,
        ),
        lm_head,
        config: config.clone(),
        cos_cache: AVec::from_vec(cos),
        sin_cache: AVec::from_vec(sin),
        gemma: Some(gemma_model),
        rope_nope: None,
        rope_sliding: None,
    };
    Ok((config, model))
}

fn out_of_range(layer: usize, config: &Config) -> HerbertError {
    HerbertError::ModelLoad(format!(
        "tensor of layer {layer} but the config declares {} layers",
        config.num_layers
    ))
}
