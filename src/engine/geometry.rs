//! What the scheduler must know about the loaded model before it compiles
//! anything — and the refusal of everything the emitters do not serve.
//!
//! What enters: the `Config` and the `Model` the loader built.
//! What comes out: the [`Geometry`] (the model shape the kernels are compiled
//! for, the layer count, the vocabulary, the GatedDeltaNet budget), the
//! distinct attention geometries, the KV plan (which layers own a cache), the
//! RoPE table sets, and the MLP widths. Nothing here copies a weight.
//!
//! The config already refuses every other architecture; what is checked here
//! is what the *emitters* need beyond the config's word: the DeltaNet tensor
//! lengths, the multiples of 16 the R16 GEMM wants, a RoPE width the rotation
//! kernel can emit, a SiLU activation. A model that fails one of these is
//! refused at load with a sentence, never served with a plausible answer.

use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::jit::{Activation, DeltaBudget, DeltaShape, ModelShape, PlanKv, RopeTableSet};
use crate::model::{GatedDeltaNet, LayerBlock, Model};

/// Retained geometry, plus what is needed to size the runner.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Shape of a layer, as expected by the scheduler.
    pub shape: ModelShape,
    pub n_layers: usize,
    pub vocab: usize,
    /// What the GatedDeltaNet mixers reserve; `None` on a model without them.
    pub delta: Option<DeltaBudget>,
}

/// Every deviation from the served perimeter, aggregated by message.
#[derive(Default)]
struct Deviations {
    /// `(message, first layer concerned, number of layers)`, in order of appearance.
    seen: Vec<(String, Option<usize>, usize)>,
}

impl Deviations {
    fn note(&mut self, layer: Option<usize>, what: &str) {
        if let Some(e) = self.seen.iter_mut().find(|e| e.0 == what) {
            e.2 += 1;
        } else {
            self.seen.push((what.to_string(), layer, 1));
        }
    }

    /// The refusal: every deviation on stderr, the first one in the error.
    fn refusal(&self, n_layers: usize) -> HerbertError {
        eprintln!(
            "[geometry] {} deviation(s) over {n_layers} layers:",
            self.seen.len()
        );
        for (what, layer, n) in &self.seen {
            match layer {
                Some(l) => eprintln!("[geometry]   {n}/{n_layers} layers (first: {l}) — {what}"),
                None => eprintln!("[geometry]   model — {what}"),
            }
        }
        let (what, layer, _) = &self.seen[0];
        let detail = match layer {
            Some(l) => format!("layer {l}: {what}"),
            None => what.clone(),
        };
        HerbertError::Backend(format!("out of the served perimeter — {detail}"))
    }
}

/// Reads the geometry and refuses anything the emitters cannot serve.
pub fn geometry(config: &Config, model: &Model) -> Result<Geometry> {
    let mut deviations = Deviations::default();
    let gemma = config.gemma.as_ref();
    // Gemma 4: the reserved maxima over the layers (the head width, the KV heads, the MLP width).
    let max_of = |f: &dyn Fn(&crate::model::DecoderLayer, usize) -> usize| {
        model
            .decoder_layers
            .iter()
            .enumerate()
            .map(|(l, d)| f(d, l))
            .max()
            .unwrap_or(0)
    };
    let attn_of = |d: &crate::model::DecoderLayer| match &d.block {
        LayerBlock::Attention(a) => Some((a.head_dim, a.num_kv_heads)),
        LayerBlock::GatedDeltaNet(_) => None,
    };
    let shape = ModelShape {
        hidden: config.hidden_size,
        n_heads: config.num_attention_heads,
        n_kv_heads: if gemma.is_some() {
            max_of(&|d, _| attn_of(d).map_or(0, |a| a.1))
        } else {
            config.num_key_value_heads
        },
        head_dim: if gemma.is_some() {
            max_of(&|d, _| attn_of(d).map_or(0, |a| a.0))
        } else {
            config.head_dim
        },
        rotary_ndims: config.rotary_ndims,
        // Without a dense MLP (Granite 3), the reserved width is the experts' (the scheduler checks
        // every activation width against it).
        intermediate: if gemma.is_some() {
            max_of(&|d, _| d.mlp.intermediate_size)
        } else {
            config
                .intermediate_size
                .max(config.qwen_moe.as_ref().map_or(0, |m| {
                    if config.intermediate_size == 0 {
                        m.inter
                    } else {
                        0
                    }
                }))
        },
        eps: config.rms_norm_eps,
        // Qwen3.5 declares no scale: `1/√head_dim`. Gemma 4 scales by one (the norms of q and k do
        // the work).
        attn_scale: gemma
            .map(|_| 1.0)
            .or(config.granite.map(|g| g.attention_multiplier)),
        // Qwen3.5 norms are zero-centred: `(1 + w) · rms(x)` for every norm of the model.
        norm_plus_one: config.norm_type == crate::core::config::NormType::GemmaRMSNorm,
        output_gate: config.attn_output_gate,
        qkv_bias: config.qkv_bias,
        gemma: gemma.map(|g| crate::jit::GemmaShape {
            pl_dim: g.pl_dim,
            n_layers: config.num_layers,
            softcap: g.final_logit_softcap,
            window: if g.layer_sliding.iter().any(|&s| s) {
                g.sliding_window
            } else {
                0
            },
            moe: g.moe.map(|m| crate::jit::MoeShape {
                experts: m.num_experts,
                top_k: m.top_k,
                inter: m.inter,
            }),
        }),
        qwen_moe: config.qwen_moe.as_ref().map(|m| crate::jit::QwenMoeShape {
            moe: crate::jit::MoeShape {
                experts: m.num_experts,
                top_k: m.top_k,
                inter: m.inter,
            },
            norm_topk: m.norm_topk_prob,
            lfm_scale: m.lfm_scale,
        }),
        deepstack_layers: config.qwen_vl.as_ref().map_or(0, |v| v.deepstack_layers),
    };
    if !config.rotary_ndims.is_multiple_of(32) {
        deviations.note(
            None,
            &format!(
                "partial RoPE not emittable (rotary_ndims={}: a multiple of 32 is expected)",
                config.rotary_ndims
            ),
        );
    }
    if config.gemma.is_none() && config.hidden_act != "silu" {
        deviations.note(
            None,
            &format!(
                "hidden_act {:?} (silu expected: SwiGLU MLP, SiLU conv)",
                config.hidden_act
            ),
        );
    }
    let mut delta: Option<DeltaBudget> = None;
    for (l, layer) in model.decoder_layers.iter().enumerate() {
        match &layer.block {
            LayerBlock::Attention(attn) => {
                let shared = layer.gemma.as_ref().is_some_and(|g| g.kv_source.is_some());
                if config.qk_norm
                    && (attn.q_norm.len() != attn.head_dim
                        || (!shared && attn.k_norm.len() != attn.head_dim))
                {
                    deviations.note(Some(l), "Q/K norm weights of unexpected length");
                }
                let (qd, kvd) = (
                    attn.num_heads * attn.head_dim,
                    attn.num_kv_heads * attn.head_dim,
                );
                if config.qkv_bias
                    && (attn.q_bias.len() != qd
                        || attn.k_bias.len() != kvd
                        || attn.v_bias.len() != kvd)
                {
                    deviations.note(Some(l), "Q/K/V biases of unexpected length");
                }
            }
            LayerBlock::GatedDeltaNet(dn) if dn.short_conv => {
                let h = config.hidden_size;
                if dn.head_v_dim != h
                    || dn.conv_kernel_size < 2
                    || dn.conv1d_weight.len() != h * dn.conv_kernel_size
                    || !h.is_multiple_of(16)
                    || dn.in_proj_c.n != h
                {
                    deviations.note(Some(l), "short convolution tensors of unexpected length");
                }
                cumulate_delta(&mut delta, dn);
            }
            LayerBlock::GatedDeltaNet(dn) => {
                deltanet_control(l, dn, config, &mut deviations);
                cumulate_delta(&mut delta, dn);
            }
        }
        if layer.mlp.intermediate_size % 16 != 0 {
            deviations.note(Some(l), "MLP width not a multiple of 16 (R16 GEMM)");
        }
        if let (Some(m), Some(e)) = (&layer.qwen_moe, &config.qwen_moe) {
            if m.shared_gate.len() != config.hidden_size || m.router.len() * 16 != e.num_experts {
                deviations.note(
                    Some(l),
                    "qwen3_5_moe: router or shared expert gate of unexpected length",
                );
            }
            if !config.hidden_size.is_multiple_of(32) {
                deviations.note(
                    Some(l),
                    "qwen3_5_moe: hidden size not a multiple of 32 (the experts' int8 rows)",
                );
            }
        }
    }
    if !deviations.seen.is_empty() {
        return Err(deviations.refusal(model.decoder_layers.len()));
    }
    Ok(Geometry {
        shape,
        n_layers: model.decoder_layers.len(),
        vocab: model.lm_head.n,
        delta,
    })
}

/// What the scheduler assumes of a GatedDeltaNet mixer: a violation would be
/// a late kernel refusal or a silent wrong computation.
fn deltanet_control(l: usize, dn: &GatedDeltaNet, config: &Config, deviations: &mut Deviations) {
    if dn.num_v_heads == 0 || !dn.num_v_heads.is_multiple_of(16) {
        deviations.note(
            Some(l),
            "DeltaNet num_v_heads not a multiple of 16 (R16 GEMM on Wa/Wb)",
        );
    }
    if dn.num_k_heads == 0 || !dn.num_v_heads.is_multiple_of(dn.num_k_heads) {
        deviations.note(
            Some(l),
            "DeltaNet GQA not integral (num_v_heads / num_k_heads)",
        );
    }
    if dn.conv_kernel_size < 2 {
        deviations.note(Some(l), "DeltaNet conv kernel < 2");
    }
    let key_dim = dn.num_k_heads * dn.head_k_dim;
    let conv_dim = 2 * key_dim + dn.num_v_heads * dn.head_v_dim;
    if dn.conv1d_weight.len() != conv_dim * dn.conv_kernel_size
        || dn.a_log.len() != dn.num_v_heads
        || dn.dt_bias.len() != dn.num_v_heads
        || dn.norm_weight.len()
            != if dn.mamba {
                dn.num_v_heads * dn.head_v_dim
            } else {
                dn.head_v_dim
            }
        || (dn.mamba && (dn.conv1d_bias.len() != conv_dim || dn.d.len() != dn.num_v_heads))
        || dn.in_proj_a.len() != dn.num_v_heads * config.hidden_size
        || dn.in_proj_b.len() != dn.num_v_heads * config.hidden_size
    {
        deviations.note(Some(l), "DeltaNet tensors of unexpected length");
    }
}

/// The maxima over the DeltaNet layers, which size the mixer buffers. A short conv (LFM2) has no
/// head and no recurrent state: its `B`, `C` and `x` (`hidden` wide each) go where the DeltaNet's
/// `z`, output and `[q | k | v]` do, its conv state holds `hidden` channels.
fn cumulate_delta(delta: &mut Option<DeltaBudget>, dn: &GatedDeltaNet) {
    let value_dim = if dn.short_conv {
        dn.head_v_dim
    } else {
        dn.num_v_heads * dn.head_v_dim
    };
    let conv_dim = 2 * dn.num_k_heads * dn.head_k_dim + value_dim;
    let rec = dn.num_v_heads * dn.head_k_dim * dn.head_v_dim;
    match delta {
        None => {
            *delta = Some(DeltaBudget {
                shape: DeltaShape {
                    head_k_dim: dn.head_k_dim,
                    conv_dim,
                    value_dim,
                    num_v_heads: dn.num_v_heads,
                    conv_kernel: dn.conv_kernel_size,
                },
                n_layers: 1,
                recurrent_stride: rec,
            })
        }
        Some(d) => {
            d.shape.head_k_dim = d.shape.head_k_dim.max(dn.head_k_dim);
            d.shape.conv_dim = d.shape.conv_dim.max(conv_dim);
            d.shape.value_dim = d.shape.value_dim.max(value_dim);
            d.shape.num_v_heads = d.shape.num_v_heads.max(dn.num_v_heads);
            d.shape.conv_kernel = d.shape.conv_kernel.max(dn.conv_kernel_size);
            d.recurrent_stride = d.recurrent_stride.max(rec);
            d.n_layers += 1;
        }
    }
}

/// The model's distinct MLP widths with their activation, in order of
/// appearance: the scheduler compiles one activation kernel per width.
pub fn mlp_widths(model: &Model) -> Vec<(Activation, usize)> {
    let mut v: Vec<(Activation, usize)> = Vec::new();
    for layer in &model.decoder_layers {
        let activation = if model.config.gemma.is_some() {
            Activation::GeluTanh
        } else {
            Activation::SwiGlu
        };
        let key = (activation, layer.mlp.intermediate_size);
        // No dense MLP (Granite 3): no activation to compile for it.
        if key.1 != 0 && !v.contains(&key) {
            v.push(key);
        }
    }
    // The experts of a Gemma 4 mixture-of-experts model activate `inter` values.
    if let Some(m) = model.config.gemma.as_ref().and_then(|g| g.moe) {
        let key = (Activation::GeluTanh, m.inter);
        if !v.contains(&key) {
            v.push(key);
        }
    }
    // Those of `qwen3_5_moe` too, with SwiGLU.
    if let Some(m) = &model.config.qwen_moe {
        let key = (Activation::SwiGlu, m.inter);
        if !v.contains(&key) {
            v.push(key);
        }
    }
    v
}

/// The distinct attention geometries `(head_dim, rotary_ndims)` of the model:
/// one set of kernels per `head_dim`, one RoPE kernel per `rotary_ndims`.
pub fn attention_geometries(config: &Config, model: &Model) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for layer in &model.decoder_layers {
        if let LayerBlock::Attention(attn) = &layer.block {
            let key = (attn.head_dim, attn.rotary_ndims);
            if !v.contains(&key) {
                v.push(key);
            }
        }
    }
    if v.is_empty() {
        v.push((config.head_dim, config.rotary_ndims));
    }
    v
}

/// The KV plan, one entry per layer: an attention layer owns a cache for its
/// `head_dim`, a GatedDeltaNet layer has none (its state is recurrent).
pub fn plan_kv(model: &Model) -> Vec<PlanKv> {
    model
        .decoder_layers
        .iter()
        .map(|layer| match &layer.block {
            // A Gemma layer that reads another layer's cache owns none.
            LayerBlock::Attention(_)
                if layer.gemma.as_ref().is_some_and(|g| g.kv_source.is_some()) =>
            {
                PlanKv::Absent
            }
            LayerBlock::Attention(attn) => PlanKv::Own(attn.head_dim),
            LayerBlock::GatedDeltaNet(_) => PlanKv::Absent,
        })
        .collect()
}

/// One set of `cos`/`sin` tables per declared `rotary_ndims`: the loader's
/// tables, which serve every attention layer.
pub fn tables_rope<'a>(
    model: &'a Model,
    geometries: &[(usize, usize)],
) -> Result<Vec<RopeTableSet<'a>>> {
    let mut v: Vec<RopeTableSet<'a>> = Vec::new();
    for &(_, rn) in geometries {
        if v.iter().any(|t| t.rotary_ndims == rn) {
            continue;
        }
        if let Some(g) = model
            .gemma
            .as_ref()
            .filter(|g| g.rope_full.0 == rn && rn != model.config.rotary_ndims)
        {
            v.push(RopeTableSet {
                rotary_ndims: rn,
                cos: &g.rope_full.1,
                sin: &g.rope_full.2,
                sliding: false,
            });
            continue;
        }
        if let Some((cos, sin)) = model.rope_nope.as_ref().filter(|_| {
            rn == crate::model::model::NOPE_ROTARY_NDIMS && rn != model.config.rotary_ndims
        }) {
            v.push(RopeTableSet {
                rotary_ndims: rn,
                cos,
                sin,
                sliding: false,
            });
            continue;
        }
        if rn != model.config.rotary_ndims {
            return Err(HerbertError::Backend(format!(
                "no RoPE tables for rotary_ndims={rn}: the loader built them for {}",
                model.config.rotary_ndims
            )));
        }
        v.push(RopeTableSet {
            rotary_ndims: rn,
            cos: &model.cos_cache,
            sin: &model.sin_cache,
            sliding: false,
        });
    }
    // LFM3: the sliding layers' tables, of the model's width, after the full layers' set.
    if let Some((cos, sin)) = &model.rope_sliding {
        v.push(RopeTableSet {
            rotary_ndims: model.config.rotary_ndims,
            cos,
            sin,
            sliding: true,
        });
    }
    Ok(v)
}
