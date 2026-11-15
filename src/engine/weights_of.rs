//! The borrow of every layer's weights, as the scheduler expects them.
//!
//! Nothing is allocated or repacked here: slices over what the loader and
//! [`crate::engine::packs`] already produced. On a GatedDeltaNet layer the six
//! attention fields receive empty slices the scheduler never reads; on an
//! attention layer the mixer carries no DeltaNet weights.

use crate::engine::packs::DeltaPacks;
use crate::jit::{DeltaWeights, LayerWeights, Mixer};
use crate::model::{LayerBlock, Model};

pub fn layer_weights<'a>(
    model: &'a Model,
    packs: &'a [Option<DeltaPacks>],
) -> Vec<LayerWeights<'a>> {
    const EMPTY: &[u16] = &[];
    model
        .decoder_layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            let mlp = &layer.mlp;
            let (attn, mixer) = match (&layer.block, packs.get(l).and_then(Option::as_ref)) {
                (LayerBlock::GatedDeltaNet(dn), Some(p)) => (
                    None,
                    Mixer::GatedDeltaNet(DeltaWeights {
                        in_proj_qkv: &dn.in_proj_qkv.data_r16,
                        in_proj_z: &dn.in_proj_z.data_r16,
                        in_proj_a: &p.in_proj_a.data_r16,
                        in_proj_b: &p.in_proj_b.data_r16,
                        conv1d_weight: &p.conv1d_weight,
                        // The only f32 of the block: `decay = exp(−exp(a_log) · dt)`.
                        a_log: &dn.a_log,
                        dt_bias: &p.dt_bias,
                        norm_weight: &dn.norm_weight,
                        out_proj: &dn.out_proj.data_r16,
                        num_k_heads: dn.num_k_heads,
                        num_v_heads: dn.num_v_heads,
                        head_k_dim: dn.head_k_dim,
                        head_v_dim: dn.head_v_dim,
                        conv_kernel: dn.conv_kernel_size,
                        mamba: dn.mamba,
                        conv1d_bias: &dn.conv1d_bias,
                        d: &dn.d,
                        short_conv: dn.short_conv,
                        in_proj_c: &dn.in_proj_c.data_r16,
                    }),
                ),
                (LayerBlock::Attention(a), _) => (Some(a), Mixer::Attention),
                (LayerBlock::GatedDeltaNet(_), None) => {
                    unreachable!("delta_packs() builds one pack per DeltaNet layer")
                }
            };
            LayerWeights {
                input_norm: &layer.input_layernorm,
                post_attn_norm: &layer.post_attention_layernorm,
                q_norm: attn.map(|a| &a.q_norm[..]).unwrap_or(EMPTY),
                k_norm: attn.map(|a| &a.k_norm[..]).unwrap_or(EMPTY),
                q_bias: attn.map(|a| &a.q_bias[..]).unwrap_or(EMPTY),
                k_bias: attn.map(|a| &a.k_bias[..]).unwrap_or(EMPTY),
                v_bias: attn.map(|a| &a.v_bias[..]).unwrap_or(EMPTY),
                wq: attn.map(|a| &a.q_proj.data_r16[..]).unwrap_or(EMPTY),
                wk: attn.map(|a| &a.k_proj.data_r16[..]).unwrap_or(EMPTY),
                wv: attn.map(|a| &a.v_proj.data_r16[..]).unwrap_or(EMPTY),
                wo: attn.map(|a| &a.o_proj.data_r16[..]).unwrap_or(EMPTY),
                activation: if model.config.gemma.is_some() {
                    crate::jit::Activation::GeluTanh
                } else {
                    crate::jit::Activation::SwiGlu
                },
                intermediate: mlp.intermediate_size,
                n_kv_heads: attn.map(|a| a.num_kv_heads),
                head_dim: attn.map_or(model.config.head_dim, |a| a.head_dim),
                rotary_ndims: attn.map_or(model.config.rotary_ndims, |a| a.rotary_ndims),
                w_gate: &mlp.gate_proj.data_r16,
                w_up: &mlp.up_proj.data_r16,
                w_down: &mlp.down_proj.data_r16,
                mixer,
                gemma: layer.gemma.as_ref().map(|g| crate::jit::GemmaLayerWeights {
                    post_attn_out_norm: &g.post_attn_out_norm,
                    post_ff_norm: &g.post_ff_norm,
                    window: g.window,
                    kv_source: g.kv_source,
                    ple: g.ple.as_ref().map(|p| crate::jit::PleLayerWeights {
                        gate: &p.gate.data_r16,
                        proj: &p.proj.data_r16,
                        norm: &p.norm,
                    }),
                    k_eq_v: g.k_eq_v,
                    moe: g.moe.as_ref().map(|m| crate::jit::MoeLayerWeights {
                        router: m.router.iter().map(|r| &r.data_r16[..]).collect(),
                        router_scale: &m.router_scale,
                        per_expert_scale: &m.per_expert_scale,
                        pre_ff_norm_2: &m.pre_ff_norm_2,
                        post_ff_norm_1: &m.post_ff_norm_1,
                        post_ff_norm_2: &m.post_ff_norm_2,
                        gate_up: m.gate_up.iter().map(|w| &w.data_r16[..]).collect(),
                        down: m.down.iter().map(|w| &w.data_r16[..]).collect(),
                    }),
                    layer_scalar: g.layer_scalar,
                }),
                qwen_moe: layer
                    .qwen_moe
                    .as_ref()
                    .map(|m| crate::jit::QwenMoeLayerWeights {
                        router: m.router.iter().map(|r| &r.data_r16[..]).collect(),
                        shared_gate: &m.shared_gate,
                        gate_up: m.gate_up.iter().map(|w| &w.data_r16[..]).collect(),
                        down: m.down.iter().map(|w| &w.data_r16[..]).collect(),
                        expert_bias: &m.expert_bias,
                    }),
                lfm3: layer.lfm3.as_ref().map(|x| crate::jit::Lfm3LayerWeights {
                    post_mixer_norm: &x.post_operator_norm,
                    post_mlp_norm: &x.post_ffn_norm,
                    head_gate: &x.head_gate.data_r16,
                    window: x.window,
                }),
            }
        })
        .collect()
}
