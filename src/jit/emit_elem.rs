//! The elementwise kernels of a layer, whatever the target: one blob per
//! `(op, dim)`.
//!
//! | kernel | site in the layer | formula |
//! |---|---|---|
//! | [`ElemOp::RmsNorm`], [`ElemOp::RmsNormGemma`] | the layer norms, the final norm | `y = x · rsqrt(Σx²/d + ε) · w` (Gemma: `· (1 + w)`) |
//! | [`ElemOp::SwiGlu`] | the MLP activation | `y = gate · σ(gate) · up` |
//! | [`ElemOp::SigmoidMul`] | the attention output gate | `y = x · σ(gate)` |
//! | [`ElemOp::ResidualAdd`] | the two residuals | `a += b` |
//! | [`ElemOp::Rope`] | the rotation of Q and K | `(x, x') ← (x·c − x'·s, x·s + x'·c)` |
//! | [`ElemOp::Clamp`] | the clipped linears of the vision tower | `y = min(max(x, lo), hi)` |
//! | [`ElemOp::PoolRows`] | the pooling of the vision tower | `y = bf16(Σᵣ x[r] · inv) · s` |
//! | [`ElemOp::Standardize`] | the standardized soft tokens (26B-A4B) | `y = (x · r − b) · c` |
//! | [`ElemOp::LayerNorm`] | the norms of XLM-RoBERTa | `y = (x − μ) · rsqrt(σ² + ε) · g + b` |
//! | [`ElemOp::GeluErf`] | the MLP of XLM-RoBERTa | `y = g · Φ(g) · up` |
//! | [`ElemOp::DepthwiseRows`] | the light convolution of the Gemma 4 audio tower | `y = bf16(Σᵣ x[r] · w[r])` |
//! | [`ElemOp::LocalAttention`] | the attention of the Gemma 4 audio tower | `softmax(cap·tanh((q'·k'ᵃ + q'·relᵃ)/cap)) · v` |
//!
//! Everything is bf16 in memory and accumulates in f32 in registers; no
//! kernel converts at its input or output — the producer writes what the
//! consumer reads. `dim` is an immediate: one shape, one kernel, no remainder
//! check at run time. The emitters: `arch/x86/emit_elem.rs` (AVX-512),
//! `arch/x86/emit_elem_avx2.rs`, `arch/aarch64/elem.rs` (NEON).

use crate::jit::ir::Decline;

/// The elementary kernels that this emitter can produce.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ElemOp {
    /// `y[i] = x[i] · rsqrt(Σx²/d + ε) · w[i]`, all in bf16.
    ///
    /// ABI: `fn(x: *const u16, w: *const u16, y: *mut u16, eps: f32)`.
    RmsNorm,
    /// `y[i] = x[i] · rsqrt(Σx²/d + ε) · (1 + w[i])` — the *Gemma* convention,
    /// which **Qwen3.5** applies to all its norms (`NormType::GemmaRMSNorm`).
    ///
    /// Same ABI as [`ElemOp::RmsNorm`], and **the same emitter**: the only difference
    /// is a `vaddps` of a broadcast constant on widened `w`, after the
    /// `vpslld` that brings it to f32. One instruction per half-block.
    ///
    /// **Do not assume we can get away with folding `w' = w + 1` at
    /// load time.** The identity holds over the reals, but is false in the format:
    /// `w'` would be stored in bf16, whose quantum near 1.0 is
    /// `2⁻⁸ ≈ 3.9e-3`, whereas these weights are small deviations around 0.
    /// A `w = 1.2e-2` would fold to within 3.9e-3 — **~30 % relative error**.
    /// The scalar reference adds in **f32**, after widening; this kernel does
    /// the same, in the same place.
    RmsNormGemma,
    /// `y[i] = gate[i] · σ(gate[i]) · up[i]`, with `σ(t) = 1/(1+e^{-t})`.
    ///
    /// ABI: `fn(gate: *const u16, up: *const u16, y: *mut u16)`.
    SwiGlu,
    /// `y[i] = geluTanh(gate[i]) · up[i]`, the Gemma4 activation.
    ///
    /// `geluTanh(g) = ½·g·(1 + tanh(z))`, `z = √(2/π)·(g + 0.044715·g³)`.
    ///
    /// **Emitted as a SwiGLU with cubic argument**, via the identity
    /// `½(1 + tanh z) = σ(2z)` :
    ///
    /// ```text
    /// geluTanh(g) = g · σ(t),   t = 2z = K·g + K·C·g³ = g·(K + K·C·g²)
    /// K = 2√(2/π) = 1.5957691   C = 0.044715
    /// ```
    ///
    /// All the exponential machinery is therefore shared with
    /// [`ElemOp::SwiGlu`]: four more instructions, not a second kernel.
    /// And the argument to `exp` remains bounded by `EXP_MAX88` as there.
    ///
    /// ABI: `fn(gate: *const u16, up: *const u16, y: *mut u16)`.
    GeluTanh,
    /// `y[i] = a[i] · σ(b[i])` — the **attention output gate** of Qwen3.5.
    ///
    /// ABI: `fn(a: *const u16, b: *const u16, y: *mut u16)`, that of
    /// [`SwiGluFn`]. `a` is the SDPA output, `b` the gate.
    ///
    /// **Emitted by the [`ElemOp::SwiGlu`] arm**, which is a variant with
    /// two differences, and not a second emitter:
    ///
    /// 1. the sigmoid argument comes from the **second** operand (`b`, `rsi`) instead of the
    ///    first;
    /// 2. the final `vmulps` **disappears** — the `vdivps` of SwiGLU already divides by `1 +
    ///    exp(−·)`, it suffices to give it `a` in the numerator: `a / (1 + exp(−b)) = a · σ(b)`.
    ///
    /// No extra register, no extra constant. In particular the
    /// bound `EXP_MAX88` is **reused**: as with SwiGLU the argument to
    /// `exp` is `−b`, positive as soon as `b` is negative, and without the bound a
    /// `b < −88` would yield an **arbitrary** number instead of ~0.
    SigmoidMul,
    /// `a[i] += b[i]`, in place.
    ///
    /// ABI: `fn(a: *mut u16, b: *const u16)`.
    ResidualAdd,
    /// `y[i] = x[i] · s`, all in bf16, `s` scalar f32.
    ///
    /// ABI: `fn(x: *const u16, y: *mut u16, s: f32)` — that of [`RmsNormFn`]
    /// **without the weights**, the scalar taking the place of the `eps`.
    ///
    /// The three scales of the Gemma4 input branch: the end-of-layer `layer_scalar`, the
    /// `√256` from the `embed_tokens_per_layer` lookup, and the `× 1/√2` from the
    /// upstream combine.
    ///
    /// Body of [`ElemOp::ResidualAdd`] with one difference: the second operand
    /// is not read from memory but broadcast once from `xmm0`, and the
    /// output goes into `rsi` instead of overwriting the input.
    Scale,
    /// RoPE on one head, in place, `rotate_half` convention (Llama/Qwen):
    /// for `i < dim/2`, the pair `(x[i], x[i+dim/2])` rotates by the angle of the
    /// pair `(cos[i], sin[i])`.
    ///
    /// ABI: `fn(x: *mut u16, cos: *const u16, sin: *const u16)`, `dim` being
    /// the `head_dim` and the tables having `dim/2` inputs.
    ///
    /// The `cos`/`sin` tables are in **bf16**, like everything in
    /// memory. They are model constants just like the **weights** — but their
    /// precision directly affects the rotation angle, unlike a **weight**.
    /// The error is **measured** instead of assumed negligible.
    Rope,
    /// `y[i] = cap · tanh(x[i] / cap)` — the softcapping of logits of the Gemma
    /// models; not used by Qwen3.5, kept as an emitted operation.
    ///
    /// ABI: that of [`ElemOp::Scale`], `fn(x: *const u16, y: *mut u16,
    /// cap: f32)` — a broadcast scalar, one input, one output.
    ///
    /// **In greedy mode, this gap is not visible**: `tanh` is monotone
    /// increasing, so the argmax is invariant. It is however false as soon as we
    /// sample (temperature, top-p): without the cap, the gap between two
    /// logits is not compressed and the distribution after softmax is not the
    /// same. A kernel that no non-regression greedy can claim.
    ///
    /// **Emitted with the exponential of [`ElemOp::SwiGlu`]**, not a second
    /// transcendental: via `tanh(u) = (1 − e^{−2u}) / (1 + e^{−2u})` applied to
    /// `|u|`, with the sign restored at the end (`vpord`). Taking `|u|` is not
    /// cosmetic — the argument of `exp` remains **negative or zero**, thus within the
    /// domain that `EXP_MAX88` guards, and the saturation `|x| ≫ cap` outputs
    /// exactly `±cap` instead of an `∞/∞`.
    Softcap,
    /// `y[i] = min(max(x[i], lo), hi)` — the input and the output bounds of a
    /// clipped linear of the Gemma 4 vision tower (`input_min`, `input_max`,
    /// `output_min`, `output_max`).
    ///
    /// ABI: `fn(x: *const u16, y: *mut u16, lo: f32, hi: f32)`, that of
    /// [`ElemOp::Scale`] with a second scalar. `x` and `y` may be one buffer.
    /// No rounding: a bf16 in, the same bf16 or a bound out (the bounds are
    /// bf16 values of the checkpoint).
    Clamp,
    /// `y[i] = bf16(bf16(Σ_{r<rows} x[r·dim + i] · inv) · s)` — the average
    /// pooling of the vision tower: `rows` consecutive rows of `dim` values
    /// (the `k × k` patches of one cell, laid out together), summed by fused
    /// multiply-adds of `inv = 1/k²` in row order from zero, rounded to bf16,
    /// then multiplied by `s = √hidden`.
    ///
    /// ABI: `fn(x: *const u16, y: *mut u16, s: f32, inv: f32)`.
    PoolRows(u8),
    /// `y[i] = bf16((x[i] · r − b[i]) · c[i])`, every step in f32, one
    /// rounding at the end: the standardization of the pooled soft tokens of
    /// a vision tower with `standardize` (Gemma 4 26B-A4B: transformers ≥ 5.20's
    /// `Gemma4VisionModel.forward`, the pooled mean in bf16 times
    /// `r = √hidden` in f32, minus `std_bias`, times `std_scale`, then cast).
    ///
    /// ABI: that of [`ElemOp::RmsNorm`], `fn(x, w, y, r)`, `w` holding the
    /// `dim` values of `b` then the `dim` values of `c`.
    Standardize,
    /// `y[i] = (x[i] − μ) · rsqrt(σ² + ε) · g[i] + b[i]`, `μ` the mean of the
    /// row and `σ²` the mean of `(x − μ)²` (torch's `LayerNorm`, the biased
    /// variance), in f32 from the bf16 row.
    ///
    /// ABI: that of [`ElemOp::RmsNorm`], `fn(x, w, y, eps)`, `w` holding the
    /// `dim` values of `g` then the `dim` values of `b`.
    LayerNorm,
    /// `y[i] = gelu(gate[i]) · up[i]` with the exact GeLU, `gelu(g) = g · Φ(g)`
    /// (`½·g·(1 + erf(g/√2))`, transformers' `gelu`): XLM-RoBERTa's MLP, which
    /// passes a row of ones as `up`.
    ///
    /// `Φ(g) = ½·erfc(−g/√2)` by the `erfcc` of Numerical Recipes (fractional
    /// error below 1.2e-7): `a = |g|/√2`, `t = 1/(1 + a/2)`,
    /// `erfc(a) = t·exp(−a² + P(t))`, `P` of degree 9, the exponential of the
    /// other kernels. ABI: that of [`ElemOp::GeluTanh`].
    GeluErf,
    /// `y[i] = bf16(Σ_{r<taps} x[r·dim + i] · w[r·dim + i])`, the products fused in row order from
    /// zero — the causal depthwise convolution of the light convolution of the Gemma 4 audio tower
    /// (after its GLU): `x` points at the first of `taps` consecutive rows (the output row's frame
    /// and the `taps − 1` before it, zero rows before the recording), `w` holds the kernel
    /// transposed, `[taps][dim]`.
    ///
    /// ABI: that of [`ElemOp::SwiGlu`], `fn(x, w, y)`.
    DepthwiseRows(u8),
    /// The chunked local attention of the Gemma 4 audio tower, one query row,
    /// every head: row `t` sees the keys
    /// `t − a`, `a < min(t + 1, window)`, each with the relative key of
    /// distance `a`; for a head,
    /// `q' = q · q_scale · sp` (f32), `l_a = (q'·k_{t−a}) · k_scale + q'·rel[window − a]`,
    /// `l_a = cap · tanh(l_a / cap)`, `p = softmax(l)` and `y = bf16(Σ_a p_a · v_{t−a})`,
    /// everything in f32 from the bf16 rows. `dim` is the head's width.
    ///
    /// ABI: `fn(q, k, v, rel, y, t, q_scale, k_scale, cap)`: `q`, `k`, `v` and
    /// `y` point at row `t` of `[rows][heads·dim]` buffers (the `window − 1`
    /// rows before row 0 of `k` and `v` readable, zeros), `rel` at `sp`
    /// (`dim` bf16, `softplus(per_dim_scale)`) followed by the relative keys
    /// `[window + 1][heads·dim]` (positions `window … 0`), `t` the row's index.
    ///
    /// With `rows > 0`, the block kernel: one task computes `rows` query rows
    /// of one head (a tile), the same values bit for bit — each `(row, key)`
    /// dot product its own accumulator over the same lanes in the same order,
    /// the same reduction, softcap, softmax and rounding — but each key and
    /// value chunk widened once for the rows of the tile that see it, and the
    /// relative keys, the same for every row, a small GEMM of the tile. ABI:
    /// [`LocalAttentionBlockFn`], `fn(call, task, q_scale, k_scale, cap)`,
    /// `task = tile · heads + head`, the tile's rows `[tile·rows, min((tile + 1)·rows,
    /// *call.rows))`; the buffers readable up to the end of the last tile (rows past
    /// `*call.rows` are read, never written).
    LocalAttention {
        /// The heads of a row.
        heads: u8,
        /// The keys a row sees at most, itself included (`attention_context_left − 1`).
        window: u8,
        /// `0`: one row, every head (the row kernel); else the rows of a tile (a multiple of 8, at
        /// most 32).
        rows: u8,
    },
}

impl ElemOp {
    /// Short name, for logs and cache.
    pub const fn name(self) -> &'static str {
        match self {
            ElemOp::RmsNorm => "rmsnorm",
            ElemOp::RmsNormGemma => "rmsnorm_gemma",
            ElemOp::SwiGlu => "swiglu",
            ElemOp::GeluTanh => "gelutanh",
            ElemOp::SigmoidMul => "sigmoid_mul",
            ElemOp::ResidualAdd => "residual",
            ElemOp::Scale => "scale",
            ElemOp::Rope => "rope",
            ElemOp::Softcap => "softcap",
            ElemOp::Clamp => "clamp",
            ElemOp::PoolRows(_) => "pool_rows",
            ElemOp::Standardize => "standardize",
            ElemOp::LayerNorm => "layernorm",
            ElemOp::GeluErf => "gelu_erf",
            ElemOp::DepthwiseRows(_) => "depthwise_rows",
            ElemOp::LocalAttention { rows: 0, .. } => "local_attention",
            ElemOp::LocalAttention { .. } => "local_attention_block",
        }
    }
}

/// Shape of a kernel element: the operation and the dimension, both
/// fixed at load time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ElemSpec {
    /// The operation.
    pub op: ElemOp,
    /// Vector length, **immediate**.
    pub dim: usize,
}

/// `fn(x, w, y, eps)` — [`ElemOp::RmsNorm`].
// The calling convention of generated code (`sysv64` on x86-64 on every
// operating system, AAPCS64 on AArch64): see `arch::generated_fn!`.
pub type RmsNormFn = crate::jit::arch::generated_fn!(*const u16, *const u16, *mut u16, f32);
/// `fn(gate, up, y)` — [`ElemOp::SwiGlu`] and [`ElemOp::GeluTanh`], same ABI.
pub type SwiGluFn = crate::jit::arch::generated_fn!(*const u16, *const u16, *mut u16);
/// `fn(a, b)` — [`ElemOp::ResidualAdd`].
pub type ResidualAddFn = crate::jit::arch::generated_fn!(*mut u16, *const u16);
/// `fn(x, y, s)` — [`ElemOp::Scale`].
pub type ScaleFn = crate::jit::arch::generated_fn!(*const u16, *mut u16, f32);
/// `fn(x, cos, sin)` — [`ElemOp::Rope`].
pub type RopeFn = crate::jit::arch::generated_fn!(*mut u16, *const u16, *const u16);
/// `fn(x, y, lo, hi)` — [`ElemOp::Clamp`].
pub type ClampFn = crate::jit::arch::generated_fn!(*const u16, *mut u16, f32, f32);
/// `fn(x, y, s, inv)` — [`ElemOp::PoolRows`].
pub type PoolRowsFn = crate::jit::arch::generated_fn!(*const u16, *mut u16, f32, f32);
/// `fn(x, w, y)` — [`ElemOp::DepthwiseRows`].
pub type DepthwiseRowsFn = crate::jit::arch::generated_fn!(*const u16, *const u16, *mut u16);
/// `fn(q, k, v, rel, y, t, q_scale, k_scale, cap)` — [`ElemOp::LocalAttention`].
pub type LocalAttentionFn = crate::jit::arch::generated_fn!(
    *const u16,
    *const u16,
    *const u16,
    *const u16,
    *mut u16,
    usize,
    f32,
    f32,
    f32
);
/// `fn(call, task, q_scale, k_scale, cap)` — [`ElemOp::LocalAttention`] with `rows > 0` (the block
/// kernel).
pub type LocalAttentionBlockFn =
    crate::jit::arch::generated_fn!(*const LocalAttentionCall, usize, f32, f32, f32);

/// The buffers of the block kernel of [`ElemOp::LocalAttention`]: the row kernel's, at row 0.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LocalAttentionCall {
    /// `[rows][heads·dim]`.
    pub q: *const u16,
    /// Row 0 of the keys, the `window − 1` rows before it readable (zeros).
    pub k: *const u16,
    /// Row 0 of the values, likewise.
    pub v: *const u16,
    /// `sp` then the relative keys, as the row kernel's.
    pub rel: *const u16,
    /// `[rows][heads·dim]`, the output.
    pub y: *mut u16,
    /// The rows of the call (the block's, `frame.rows`).
    pub rows: *const usize,
}

impl ElemSpec {
    /// Emissible shapes.
    ///
    /// `dim` multiple of 32: this is what a zmm holds in bf16, and what makes
    /// the loops remainder-free. The two dimensions of Qwen3-0.6B (1024, 3072)
    /// and those of Qwen3.5 (1024, 3584) are. The operations of
    /// [`Self::tails`] also take an even `dim` with a partial last block (the
    /// vision towers of Gemma 4 26B-A4B and Qwen3.5: heads of 72, so a RoPE of
    /// 36 per axis, an MLP of 4 304); an emitter that does not serve that tail
    /// checks [`Self::check_whole`] instead.
    pub fn check(&self) -> Result<(), Decline> {
        if self.dim == 0 {
            return Err(Decline::Empty);
        }
        if !self.dim.is_multiple_of(32) && !(self.tails() && self.dim.is_multiple_of(2)) {
            return Err(Decline::PartialRowBlock);
        }
        match self.op {
            ElemOp::PoolRows(0) | ElemOp::DepthwiseRows(0) => return Err(Decline::Empty),
            // The logits of a row sit in sixteen lanes.
            ElemOp::LocalAttention { heads, window, .. }
                if heads == 0 || window == 0 || window > 16 =>
            {
                return Err(Decline::Empty)
            }
            // The block kernel's tiles: groups of eight rows.
            ElemOp::LocalAttention { rows, .. } if rows % 8 != 0 || rows > 32 => {
                return Err(Decline::Empty)
            }
            _ => {}
        }
        Ok(())
    }

    /// 32 bf16 vectors covered by `dim`.
    pub const fn blocks32(&self) -> usize {
        self.dim / 32
    }

    /// The operations whose AVX-512 emitter serves a partial last block (masked loads and stores).
    pub const fn tails(&self) -> bool {
        matches!(
            self.op,
            ElemOp::RmsNorm | ElemOp::Rope | ElemOp::GeluTanh | ElemOp::Standardize
        )
    }

    /// [`Self::check`] and `dim` a multiple of 32: what an emitter without the partial last block
    /// serves.
    pub fn check_whole(&self) -> Result<(), Decline> {
        self.check()?;
        if !self.dim.is_multiple_of(32) {
            return Err(Decline::PartialRowBlock);
        }
        Ok(())
    }
}

/// The two emission arms, as seen by the loop policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopArm {
    /// AVX-512: 32 bf16 per block.
    Zmm,
    /// AVX2: 16 bf16 per block.
    Ymm,
}

/// Width from which the MLP activations and the output gate are emitted as
/// a loop: on Qwen3.5-4B the `SwiGlu` 9 216 is called once per layer in
/// decode and its unrolled body (94 KiB) arrives cold in L1i every time;
/// looping it takes 1,0 % off the decode step and the generated program
/// shrinks from 7.75 to 4.76 MB. `Knobs::elem_loop_min` overrides it.
pub const LOOP_MIN_DIM: usize = 4_096;

/// The threshold applied ([`crate::jit::knobs::Knobs::elem_loop_min`]),
/// constant for the life of the kernel cache.
pub fn loop_min_dim() -> usize {
    crate::jit::knobs::knobs().elem_loop_min
}

/// Loop or unroll: the policy, as a table.
///
/// Every elementary kernel is emitted fully unrolled: one copy of the block
/// body per 32 (or 16) elements. On the MLP width of a 27B model that is a
/// 183 KiB blob in AVX-512 and 443 KiB in AVX2, several times the L1i. The
/// looped form keeps `U` block bodies and runs `dim / (U · lanes)` turns over
/// moving cursors; the body itself is byte-for-byte the same, so the output
/// bits are the same.
///
/// The table below is a measurement, not a rule. Ten widths × eight
/// operations × `U ∈ {4, 8, 16, 32}` × five machines × three repetitions,
/// synthetic data, zero output divergence. Gain against the unrolled
/// control **at its fastest repetition** (the unrolled blob moves by 14 to
/// 18 % between runs depending on where its pages land; the loop by less
/// than 1 %):
///
/// | form | Zen 4 | Zen 5 | Lunar Lake | Zen 3 (a) | Zen 3 (b) |
/// |---|---:|---:|---:|---:|---:|
/// | `swiglu` 17 408 | +2.8 | +2.4 | +2.8 | +1.0 | −0.1 |
/// | `swiglu` 12 288 | −0.3 | +0.3 | +1.9 | −4.2 | +1.9 |
/// | `swiglu` 6 144 | 0.0 | +0.3 | −0.1 | −3.1 | −0.8 |
///
/// The norms, the residual add, the scale and RoPE only gain above
/// 8 192 elements, which no served `hidden` or `head_dim` reaches, and RoPE
/// loses up to 31 % at small widths on Zen 3: they stay unrolled. The
/// "code bytes per tick" predictor was tested and does not hold: it is a
/// constant per (machine, op) and cannot select a width.
///
/// **The cache key does not change.** `ElemSpec { op, dim }` is the whole
/// key and this function is pure in `(op, dim, arm)`, the arm being fixed
/// for the process: two callers asking the same `(op, dim)` always get the
/// same blob. Do not add a field for this.
///
/// Returns the number of blocks per loop turn, or `None` to unroll.
pub fn loop_unroll(op: ElemOp, dim: usize, arm: LoopArm) -> Option<usize> {
    if !matches!(
        op,
        ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul | ElemOp::GeluErf
    ) {
        return None;
    }
    if dim < loop_min_dim() {
        return None;
    }
    let (lanes, unroll) = match arm {
        LoopArm::Zmm => (32, 16),
        LoopArm::Ymm => (16, 8),
    };
    let blocks = dim / lanes;
    blocks.is_multiple_of(unroll).then_some(unroll)
}

/// Generates the kernel of `spec` for the target of this build (the arm it
/// selects: [`crate::jit::elem::select_isa`]).
pub fn emit_elem(spec: &ElemSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::arch::emit_elem(spec)
}
