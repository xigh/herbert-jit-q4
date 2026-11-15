//! `impl Scheduler`: the setup. Everything that compiles a kernel happens
//! here, and it happens once, at mount; a pass only calls compiled kernels.

use super::*;

impl Scheduler {
    /// Compile all kernels and reserve all buffers. **Nothing will be
    /// allocated afterwards** — this is the contract of this constructor.
    ///
    /// `geometries` = the distinct pairs `(head_dim, rotary_ndims)` in any
    /// order. A homogeneous model passes only one; Gemma4 passes two
    /// (256/256 on sliding layers, 512/512 on full layers). The `head_dim`
    /// must all be ≤ `shape.head_dim`, which sizes the buffers.
    ///
    /// It produces **one set of kernels per `head_dim`** (`HeadKernels`) and
    /// **one RoPE kernel per `rotary_ndims`**: these are two axes, not one.
    pub fn with_geometries(
        shape: ModelShape,
        budget: Budget,
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
    ) -> Result<Self> {
        Self::ascend(shape, budget, widths, geometries, None)
    }

    /// Compile a hybrid model with variable attention geometry and
    /// GatedDeltaNet mixer buffers.
    pub fn with_geometries_delta(
        shape: ModelShape,
        budget: Budget,
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
        delta: DeltaShape,
    ) -> Result<Self> {
        Self::ascend(shape, budget, widths, geometries, Some(delta))
    }

    /// The common body of the two constructors.
    fn ascend(
        shape: ModelShape,
        budget: Budget,
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
        delta: Option<DeltaShape>,
    ) -> Result<Self> {
        if geometries.is_empty() {
            return Err(HerbertError::Backend(
                "backend_jit scheduler : no attention geometry declared  (head_dim, \
                 rotary_ndims pairs)"
                    .to_string(),
            ));
        }
        for &(hd, rn) in geometries {
            // A larger `rotary_ndims` than the head would cause the RoPE kernel
            // to overflow **onto the next head**, without error or NaN. Stating
            // this here, once, rather than assuming it on every table read.
            if rn == 0 || rn > hd {
                return Err(HerbertError::Backend(format!(
                    "backend_jit scheduler: rotary_ndims={rn} out of [1, head_dim={hd}]"
                )));
            }
            // The `q`/`k`/`v`/`attn` buffers and the KV stride are sized based
            // on `shape.head_dim`: a wider layer would write out of bounds.
            // Same control as on `intermediate`, same message.
            if hd > shape.head_dim {
                return Err(HerbertError::Backend(format!(
                    "backend_jit scheduler: head_dim={hd} exceeds the reserved maximum \
                     ({}).",
                    shape.head_dim
                )));
            }
        }
        if let Some(&(_, too_many)) = widths.iter().find(|&&(_, d)| d > shape.intermediate) {
            return Err(HerbertError::Backend(format!(
                "backend_jit scheduler : intermediate={too_many} exceeds the reserved \
                 maximum ({})",
                shape.intermediate
            )));
        }
        let need = |op: ElemOp, dim: usize| -> Result<Arc<CompiledElem>> {
            elem_kernel(ElemSpec { op, dim })?.ok_or_else(|| {
                HerbertError::Backend(format!(
                    "backend_jit scheduler: {} dim={dim} not schedulable",
                    op.name()
                ))
            })
        };
        // A single norm convention choice for **all** norms in the model:
        // this is what `norm_plus_one` guarantees, and this is what prevents
        // processing only a part of them.
        let norm_op = if shape.norm_plus_one {
            ElemOp::RmsNormGemma
        } else {
            ElemOp::RmsNorm
        };
        // ── one set of kernels per distinct `head_dim` ────────────────────────
        //
        // Compiled **at mount time**, never during a pass: lazy compilation
        // would allocate in the middle of prefill.
        let mut heads: Vec<HeadKernels> = Vec::with_capacity(geometries.len());
        for &(hd, rn) in geometries {
            if heads.iter().any(|k| k.head_dim == hd) {
                continue;
            }
            // The shape **of this geometry** : `q_dim` and `q_row_stride`
            // derive from it, and they are the ones that parameterize the
            // block kernel.
            let sl = shape.for_layer(hd, rn, shape.n_kv_heads);
            let sdpa = head_kernel(SdpaShape {
                head_dim: hd,
                sink: false,
            })?
            .ok_or_else(|| {
                HerbertError::Backend(format!(
                    "backend_jit scheduler: sdpa head_dim={hd} not schedulable"
                ))
            })?;
            // The prefill kernel only makes sense beyond a single row. At `m = 1`
            // it would do exactly what the decode kernel does, plus extra code.
            // `sdpa_rows() == 0` disables it: the control arm of a
            // measurement. Gemma 4 takes it too (`lower_gemma_layer`): its
            // full layers (head 512) causal, its sliding ones (head 256) with
            // the window compiled from this one.
            let sdpa_prefill = if budget.max_batch > 1 && sdpa_rows() > 0 {
                prefill_kernel(
                    SdpaShape {
                        head_dim: hd,
                        sink: false,
                    },
                    sdpa_rows(),
                    // Two distinct steps: `q` is striped twice as wide as
                    // `attn` on a gated model (the gate after `Q` on each row).
                    sl.q_row_stride(),
                    sl.q_dim(),
                    budget.kv_reserve_tokens,
                )?
            } else {
                None
            };
            heads.push(HeadKernels {
                head_dim: hd,
                norm_head: need(norm_op, hd)?,
                // Compiled only if the model requires it.
                gate_row: if shape.output_gate {
                    Some(need(ElemOp::SigmoidMul, sl.q_dim())?)
                } else {
                    None
                },
                bias: if shape.qkv_bias {
                    Some((
                        need(ElemOp::ResidualAdd, sl.q_dim())?,
                        need(ElemOp::ResidualAdd, sl.kv_dim())?,
                    ))
                } else {
                    None
                },
                sdpa,
                sdpa_prefill,
            });
        }
        // ── a RoPE kernel per distinct `rotary_ndims` ───────────────────────
        //
        // `rotary_ndims`, not `head_dim`: the kernel only writes its first
        // `spec.dim` u16 values, so **the partial RoPE is this very kernel**,
        // compiled shorter. The NoPE tail is not "preserved", it is never
        // reached. The two axes often overlap but do not coincide: Qwen3.5
        // rotates 64 out of 256, Gemma4 rotates 512 out of 512.
        let mut ropes: Vec<(usize, Arc<CompiledElem>)> = Vec::with_capacity(geometries.len());
        for &(_, rn) in geometries {
            if ropes.iter().any(|(r, _)| *r == rn) {
                continue;
            }
            ropes.push((rn, need(ElemOp::Rope, rn)?));
        }
        // The SDPA kernel pads scores to the next multiple of 16. During prefill,
        // the blocks (head, chunk) run in parallel: one slot per worker,
        // otherwise two blocks would write to the same buffer. One slot holds the
        // `rows` lines of the block — the **maximum** across the datasets, since
        // a `head_dim` may have a block kernel and another may not.
        //
        // A slot is the scratch of the block kernel (`CompiledSdpaPrefill::
        // scratch_len`: its rows of scores on x86; on AArch64 the scores, `P`
        // and the packed queries of its two GEMMs, and the queries and the
        // output of a partial tile), or one row of scores for the decode kernel.
        //
        // A sliding layer takes the block kernel of its window, compiled from
        // this one in the lowering: on x86 its scratch is up to sixteen
        // columns a row longer than the causal one's when the cache holds
        // less than `window − 1 + rows` positions (a short `--ctx`). The slot
        // covers the longest of them, the window of the whole cache.
        let scores_stride = heads
            .iter()
            .map(|k| {
                k.sdpa_prefill.as_ref().map_or(
                    budget.kv_reserve_tokens.next_multiple_of(16).max(16),
                    |p| {
                        let s = p.spec();
                        p.scratch_len().max(
                            crate::jit::emit_sdpa::SdpaPrefillSpec {
                                window: s.kv_capacity.max(1),
                                full: false,
                                ..s
                            }
                            .scratch_len(),
                        )
                    },
                )
            })
            .max()
            .unwrap_or(16);
        // A model that works one row at a time (the mixture of experts) still runs its attention
        // heads on every worker.
        let slots = if budget.max_batch > 1 || shape.experts().is_some() {
            global_pool().num_workers().max(1)
        } else {
            1
        };
        let mut s = Self {
            heads,
            ropes,
            norm_hidden: need(norm_op, shape.hidden)?,
            swiglu: {
                let mut v = Vec::with_capacity(widths.len());
                for &(act, dim) in widths {
                    let op = match act {
                        Activation::SwiGlu => ElemOp::SwiGlu,
                        Activation::GeluTanh => ElemOp::GeluTanh,
                    };
                    v.push((act, dim, need(op, dim)?));
                }
                // The gate of the per-layer input branch: `gelu(gate) · input`, `pl_dim` wide.
                if let Some(g) = shape.gemma.filter(|g| g.pl_dim > 0) {
                    if !v
                        .iter()
                        .any(|&(a, d, _)| a == Activation::GeluTanh && d == g.pl_dim)
                    {
                        v.push((
                            Activation::GeluTanh,
                            g.pl_dim,
                            need(ElemOp::GeluTanh, g.pl_dim)?,
                        ));
                    }
                }
                v
            },
            residual: need(ElemOp::ResidualAdd, shape.hidden)?,
            recurrence: None,
            conv: None,
            gated_norm: None,
            gemma: match shape.gemma {
                Some(g) => Some(GemmaKernels {
                    scale_hidden: need(ElemOp::Scale, shape.hidden)?,
                    ple: if g.pl_dim > 0 {
                        Some(PleKernels {
                            norm_pl: need(norm_op, g.pl_dim)?,
                            scale_ple: need(ElemOp::Scale, g.n_layers * g.pl_dim)?,
                            add_ple: need(ElemOp::ResidualAdd, g.n_layers * g.pl_dim)?,
                        })
                    } else {
                        None
                    },
                }),
                None => None,
            },
            ones: crate::core::avec::AVec::from_vec(vec![0x3f80u16; shape.head_dim]),
            zeros: crate::core::avec::AVec::from_vec(if shape.qwen_moe.is_some() {
                vec![0u16; budget.max_batch.max(1) * shape.hidden]
            } else {
                Vec::new()
            }),
            deepstack: crate::core::avec::AVec::from_vec(vec![
                0u16;
                shape.deepstack_layers
                    * budget.max_batch.max(1)
                    * shape.hidden
            ]),
            buffers: Buffers::new(shape, budget, scores_stride, slots),
            max_batch: budget.max_batch.max(1),
            scores_stride,
            shape,
        };
        if let Some(delta) = delta {
            let m = budget.max_batch.max(1);
            // The only allocation of the mixer, and it happens **here**: the
            // contract of these constructors is that nothing is allocated
            // afterwards.
            s.buffers.dn_qkv = crate::core::avec::AVec::zeroed(m * delta.conv_dim);
            s.buffers.dn_z = crate::core::avec::AVec::zeroed(m * delta.value_dim);
            s.buffers.dn_ab = crate::core::avec::AVec::zeroed(m * 2 * delta.num_v_heads);
            s.buffers.dn_out = crate::core::avec::AVec::zeroed(m * delta.value_dim);
            // The f32 → bf16 boundary before `Wo`, the only u16 buffer of the
            // mixer.
            s.buffers.dn_out_bf16 = crate::core::avec::AVec::zeroed(m * delta.value_dim);
            // The two gates, `max_batch · num_v_heads` each: they are
            // computed per token.
            s.buffers.dn_beta = crate::core::avec::AVec::zeroed(m * delta.num_v_heads);
            s.buffers.dn_decay = crate::core::avec::AVec::zeroed(m * delta.num_v_heads);
            // The emitted recurrence kernel, compiled **here** like all the
            // others — `None` = the scalar body (no AVX-512, or a width out of
            // budget).
            s.recurrence = recurrence_at_compile_time(&delta)?;
            // The conv + SiLU kernel emitted, compiled **here** too.
            s.conv = conv_at_compile_time(&delta)?;
            // The gated norm kernel emitted, same.
            s.gated_norm = gated_norm_at_compile_time(&delta)?;
        }
        // The packing buffer of the matrix engine, when the target's plan for
        // a block of `max_batch` rows asks for it: sized for the widest input
        // of a projection.
        let m = budget.max_batch.max(1);
        let widest_k = [
            shape.hidden,
            shape.q_dim(),
            shape.intermediate,
            delta.map_or(0, |d| d.value_dim),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let probe = GemmShape {
            k: shape.hidden,
            n: shape.hidden,
        };
        let plan = crate::jit::arch::gemm_plan(probe, m);
        if plan.opts.matrix || plan.opts.neon.w_int8 {
            // A one-row expert block (the decode, or `--moe-row-prefill`, where `max_batch` is 1)
            // packs the `top_k` rows of its experts' `down` input, `top_k · inter`
            // wide: wider than one row of the widest projection on Qwen3-VL-30B-A3B (8
            // × 768 against 4 096). A block of rows already reserves more.
            let experts = shape
                .experts()
                .map_or(0, |e| crate::jit::int8::a_bytes(e.top_k, e.inter));
            s.buffers.packed = crate::core::avec::AVec::zeroed(
                crate::jit::emit::packed_bytes(plan.opts, m, widest_k)
                    .max(experts.next_multiple_of(2))
                    / 2,
            );
        }
        Ok(s)
    }

    /// The shape served.
    pub fn shape(&self) -> ModelShape {
        self.shape
    }

    /// Number of tokens the buffers can carry in a single prefill pass.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Buffers — the caller writes the residual input stream to them and
    /// reads the output from them.
    pub fn buffers(&mut self) -> &mut Buffers {
        &mut self.buffers
    }

    /// Qwen3-VL: the deepstack vectors of an image block, `[deepstack_layers][max_batch][hidden]`
    /// bf16, row `r` of layer `l` added to row `r` of the block after layer `l`
    /// (`hidden[visual_pos_masks] += deepstack[l]`, rounded to bf16 as transformers does). The
    /// vision writes them before running the block; empty on a model without deepstack.
    pub fn deepstack_mut(&mut self) -> &mut [u16] {
        &mut self.deepstack
    }

    /// Bytes reserved by buffers.
    pub fn reserved_bytes(&self) -> usize {
        self.buffers.bytes()
    }
}
