//! A weight matrix packed the way the emitted GEMM reads it: the R16 layout.
//!
//! What enters: a row-major bf16 matrix `[N, K]` straight from the checkpoint.
//! What comes out: the same values, reordered as
//! `[N/16 blocks, K_even/2 pairs, 16 rows, 2 values]`, so that one 64-byte load
//! brings a pair of K values for sixteen output rows and `vdpbf16ps` (or its
//! AVX2 emulation) multiplies it against a broadcast pair of activations.
//! Rows past N in the last block and the column past an odd K are zero.
//! In bf16 no value changes: bf16 in, the same bf16 out. The other forms share the struct: f16
//! (`w_f16`, AArch64 without BF16), and int8 or 4-bit groups with f32 scales (`w_int8`, `w_q4`;
//! [`crate::jit::int8::pack`], the bytes held in the `u16` vector) for the shapes
//! [`crate::jit::int8::gemm`] accepts.

use crate::core::error::{HerbertError, Result};
use crate::core::hugepages::HugeVec;
use crate::core::tensor::BF16;

pub const ROWS_INTERLEAVED: usize = crate::jit::arch::R16_ROWS;

/// `R16` row-interleaved bf16 weight matrix: each pair is stored as
/// `[w(k0,row0), w(k1,row0), w(k0,row1), w(k1,row1), ...]`.
#[derive(Clone)]
pub struct PackedBf16R16Weight {
    /// Packed data in `[N/16 blocks, K_even/2 pairs, 16 rows, 2]` order.
    pub data_r16: HugeVec<BF16>,
    /// Number of output features (rows).
    pub n: usize,
    /// Number of input features (columns).
    pub k: usize,
    /// K rounded up to even: the kernel consumes pairs.
    pub k_even: usize,
    /// Number of row blocks (= ceil(N / 16)).
    pub n_blocks: usize,
}

impl PackedBf16R16Weight {
    /// A weight of shape `n×k` **without data** — only for a tensor the model
    /// declares and never reads (an absent optional projection).
    pub fn empty(n: usize, k: usize) -> Self {
        let k_even = (k + 1) & !1;
        Self {
            data_r16: HugeVec::from_vec(Vec::new()),
            n,
            k,
            k_even,
            n_blocks: n.div_ceil(ROWS_INTERLEAVED),
        }
    }

    /// `[a; b]`: the rows of `a` then those of `b`, from their packed forms. Every form (bf16, f16,
    /// int8, Q4) is blocks of rows, one after the other, so the two block runs are put end to
    /// end: `a` must end on a whole block and both must have been packed in the same form (the
    /// same `k`, a form chosen alike for both widths).
    pub fn stack(a: &Self, b: &Self) -> Result<Self> {
        let n = a.n + b.n;
        let words = |w: &Self| {
            if crate::jit::int8::gemm(w.n, w.k) {
                crate::jit::int8::weight_len(w.n, w.k)
            } else {
                w.n_blocks * w.k_even * ROWS_INTERLEAVED
            }
        };
        if a.k != b.k
            || !a.n.is_multiple_of(ROWS_INTERLEAVED)
            || !a.n.is_multiple_of(16)
            || crate::jit::int8::gemm(a.n, a.k) != crate::jit::int8::gemm(n, a.k)
            || crate::jit::int8::gemm(b.n, b.k) != crate::jit::int8::gemm(n, a.k)
            || crate::jit::int8::q4_for(a.n) != crate::jit::int8::q4_for(n)
            || crate::jit::int8::q4_for(b.n) != crate::jit::int8::q4_for(n)
            || a.data_r16.len() != words(a)
            || b.data_r16.len() != words(b)
        {
            return Err(HerbertError::ModelLoad(format!(
                "R16 stack: [{}, {}] on [{}, {}] cannot be put end to end",
                a.n, a.k, b.n, b.k
            )));
        }
        let mut data = Vec::with_capacity(a.data_r16.len() + b.data_r16.len());
        data.extend_from_slice(&a.data_r16);
        data.extend_from_slice(&b.data_r16);
        Ok(Self {
            data_r16: HugeVec::from_vec(data),
            n,
            k: a.k,
            k_even: a.k_even,
            n_blocks: n.div_ceil(ROWS_INTERLEAVED),
        })
    }

    /// Pack from row-major into the R16 layout the emitted GEMM consumes.
    pub fn pack_from_row_major_r16_only(data: &[BF16], n: usize, k: usize) -> Result<Self> {
        let expected = n.checked_mul(k).ok_or_else(|| {
            HerbertError::ModelLoad(format!("R16 pack: N*K overflows for N={n} K={k}"))
        })?;
        if data.len() != expected {
            return Err(HerbertError::ModelLoad(format!(
                "R16 pack: {} elements for N={n} K={k}, expected {expected}",
                data.len()
            )));
        }
        if crate::jit::int8::gemm(n, k) {
            // `--int8`: int8 groups with f32 scales (`jit/int8.rs`); the u16 vector holds the
            // bytes.
            let bytes = crate::jit::int8::pack(data, n, k);
            let words: Vec<u16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            return Ok(Self {
                data_r16: HugeVec::from_vec(words),
                n,
                k,
                k_even: k,
                n_blocks: n.div_ceil(ROWS_INTERLEAVED),
            });
        }
        Self::pack_bf16(data, n, k)
    }

    /// [`Self::pack_from_row_major_r16_only`] in bf16 whatever `--int8` says (the f16 form under
    /// `--f16`, as the bf16 GEMM reads it): the vision tower's weights, which
    /// [`crate::jit::arch::gemm_plan_bf16`] multiplies.
    pub fn pack_bf16(data: &[BF16], n: usize, k: usize) -> Result<Self> {
        if n.checked_mul(k) != Some(data.len()) {
            return Err(HerbertError::ModelLoad(format!(
                "R16 pack: {} elements for N={n} K={k}",
                data.len()
            )));
        }
        let k_even = k
            .checked_add(1)
            .map(|value| value & !1)
            .ok_or_else(|| HerbertError::ModelLoad(format!("R16 pack: K={k} cannot be padded")))?;
        let n_blocks = n.div_ceil(ROWS_INTERLEAVED);
        let packed_len = n_blocks
            .checked_mul(k_even)
            .and_then(|length| length.checked_mul(ROWS_INTERLEAVED))
            .ok_or_else(|| {
                HerbertError::ModelLoad(format!(
                    "R16 pack: packed length overflows for N={n} K={k}"
                ))
            })?;
        let mut data_r16 = vec![0u16; packed_len];
        // `--f16` (AArch64 without BF16): f16 values, k-major inside each pair of a block.
        let f16 = cfg!(target_arch = "aarch64") && crate::jit::knobs::knobs().neon.w_f16;
        for block in 0..n_blocks {
            let nr = ROWS_INTERLEAVED.min(n - block * ROWS_INTERLEAVED);
            for kp in 0..(k_even / 2) {
                let k0 = kp * 2;
                let k1 = k0 + 1;
                let base = (block * (k_even / 2) + kp) * ROWS_INTERLEAVED * 2;
                for lane in 0..nr {
                    let row = block * ROWS_INTERLEAVED + lane;
                    let (v0, v1) = (
                        if k0 < k { data[row * k + k0] } else { 0 },
                        if k1 < k { data[row * k + k1] } else { 0 },
                    );
                    if f16 {
                        data_r16[base + lane] = bf16_to_f16(v0)?;
                        data_r16[base + ROWS_INTERLEAVED + lane] = bf16_to_f16(v1)?;
                    } else {
                        data_r16[base + lane * 2] = v0;
                        data_r16[base + lane * 2 + 1] = v1;
                    }
                }
            }
        }
        Ok(Self {
            data_r16: HugeVec::from_vec(data_r16),
            n,
            k,
            k_even,
            n_blocks,
        })
    }
}

/// A bf16 weight as f16 bits, rounded to nearest even. A finite value past the f16 range (65504) is
/// refused: it would become an infinity in every product it enters. Subnormal f16 values keep fewer
/// bits (below 6e-5), an absolute error under 3e-8.
fn bf16_to_f16(bits: u16) -> Result<u16> {
    let x = f32::from_bits(u32::from(bits) << 16);
    let h = half::f16::from_f32(x);
    if x.is_finite() && h.is_infinite() {
        return Err(HerbertError::ModelLoad(format!(
            "f16 weights: {x} is beyond the f16 range"
        )));
    }
    Ok(h.to_bits())
}
