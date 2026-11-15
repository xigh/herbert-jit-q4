//! int8 weights and activations (AArch64, FEAT_DotProd): the layouts and the shape rule shared by
//! the packer (`model/weight.rs`), the plan, the lowering and the kernels
//! (`arch/aarch64/gemm_int8.rs`, the quantizer of `arch/aarch64/execution/int8.rs`).
//!
//! * **Weights.** The R16 blocks of sixteen rows, each cut in groups of [`GROUP`] = 32 values of
//!   `k`. A group of a block is [`GROUP_BYTES`] = 576 bytes: eight steps of `k` by four, each step
//!   four quads of sixteen bytes (quad `c` = rows `4c..4c+3`, the four `k` of the step for each
//!   row, so that one `sdot` by element gives four rows), then the sixteen f32 scales of the rows
//!   (`absmax / 127` of the 32 values of the row). A block is `K/32` groups.
//! * **Q4 weights** (`--q4`, `NeonOptions::w_q4`): the same blocks and groups with 4-bit codes. A
//!   group of a block is [`Q4_GROUP_BYTES`] = 320 bytes: four steps of eight `k`, each step four
//!   quads of sixteen bytes (quad `c` = rows `4c..4c+3`, four bytes per row; byte `j` holds the
//!   code of `k = 8t+j` in its low nibble and that of `k = 8t+4+j` in its high nibble, two's
//!   complement), then the sixteen f32 scales of the rows. The kernel (`gemm_q4.rs`) gets `16·code`
//!   as a signed byte by `shl 4` (low) or `and 0xf0` (high), so the stored scale is `scale / 16`.
//!   By default the grid of `Q4_0` (`q4_grid8`): `scale = signed_max / -8`, codes `round(w /
//!   scale)` in `[-8, 7]`; under `--q4-grid7`, `scale = absmax / 7`, codes in `[-7, 7]`.
//! * **Activations.** Per row, `K` int8 then `K/32` f32 scales (`absmax / 127` of each group of
//!   32): a row is [`a_row_bytes`] bytes, the layout the quantizing node writes and the kernels
//!   read.
//! * **Product.** `C[i][j] = Σ_g  scale_a[i][g] · scale_w[j][g] · Σ_{k in g} qa[i][k] · qw[j][k]`,
//!   the inner sum in int32 by `sdot`, the rest in f32.

/// Values of `k` per scale.
pub const GROUP: usize = 32;
/// x86-64 (`vpdpbusd`: unsigned weights × signed activations): the weight codes are offset to
/// unsigned (int8: `q + 128`, Q4: `q + 8`, scale not divided by 16), and a group of an activation
/// row carries, after its f32 scale, the int32 sum of its codes, from which the kernel removes the
/// offset (`offset · sum`).
pub const X86: bool = cfg!(target_arch = "x86_64");
/// Bytes of metadata per group of a row of quantized activations: the f32 scale, and on x86-64 the
/// int32 sum.
pub const A_META: usize = if X86 { 8 } else { 4 };
/// Bytes of one group of one R16 block: 512 of int8 and sixteen f32 scales.
pub const GROUP_BYTES: usize = 16 * GROUP + 16 * 4;
/// The same with 4-bit codes: 256 bytes and sixteen f32 scales.
pub const Q4_GROUP_BYTES: usize = 16 * GROUP / 2 + 16 * 4;

/// Are the weights of a projection of `n` rows 4-bit (`--q4`; the head stays int8 with
/// `--q4-head-int8`)?
pub fn q4_for(n: usize) -> bool {
    let neon = crate::jit::knobs::knobs().neon;
    neon.w_q4 && !(neon.q4_head_int8 && n > 100_000)
}

/// Bytes of one group of one R16 block.
pub const fn group_bytes(q4: bool) -> usize {
    if q4 {
        Q4_GROUP_BYTES
    } else {
        GROUP_BYTES
    }
}

/// Does a projection of `n` rows and `k` columns take the int8 path? Only when `--int8` asks, `k`
/// is a multiple of [`GROUP`] and `n` is not one of the tiny gate matrices (`in_proj_a`,
/// `in_proj_b`: a few tens of rows).
pub fn gemm(n: usize, k: usize) -> bool {
    let neon = crate::jit::knobs::knobs().neon;
    isa_ok(neon.w_sse42) && neon.w_int8 && k.is_multiple_of(GROUP) && n >= 64
}

/// x86-64: can this machine run `gemm_int8.rs` (AVX-512 VNNI, AVX-VNNI or AVX2 + FMA)? `false`
/// elsewhere.
pub fn x86_kernel_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::jit::arch::x86::gemm_int8::available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Does this machine have an int8 / Q4 kernel (the default kernel, not `--q4-sse42`)?
pub fn available() -> bool {
    isa_ok(false)
}

/// AArch64 (`sdot`, FEAT_DotProd), or x86-64 with AVX-512 VNNI, AVX-VNNI or AVX2 + FMA
/// (`gemm_int8.rs`) — or, when `sse42` is forced, x86-64 with SSE4.2.
fn isa_ok(sse42: bool) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = sse42;
        crate::core::isa::feature("dotprod")
    }
    #[cfg(target_arch = "x86_64")]
    {
        if sse42 {
            crate::core::isa::feature("sse4.2")
        } else {
            x86_kernel_available()
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = sse42;
        false
    }
}

/// Bytes of one row of quantized activations of `k` values.
pub const fn a_row_bytes(k: usize) -> usize {
    k + k / GROUP * A_META
}

/// Bytes of `rows` rows of quantized activations.
pub const fn a_bytes(rows: usize, k: usize) -> usize {
    rows * a_row_bytes(k)
}

/// Bytes of one R16 block of `k` values per row.
pub const fn block_bytes(k: usize, q4: bool) -> usize {
    k / GROUP * group_bytes(q4)
}

/// Elements (u16) the packed weights of an `n × k` projection hold, in the int8 form or the
/// bf16/f16 one.
pub fn weight_len(n: usize, k: usize) -> usize {
    let blocks = n.div_ceil(16);
    if gemm(n, k) {
        blocks * block_bytes(k, q4_for(n)) / 2
    } else {
        blocks * k.div_ceil(2) * 32
    }
}

/// Quantize a row-major bf16 matrix `[n, k]` (`k` a multiple of 32) into the weight layout above,
/// as bytes.
pub fn pack(data: &[u16], n: usize, k: usize) -> Vec<u8> {
    if q4_for(n) {
        return pack_q4(data, n, k);
    }
    pack_int8(data, n, k)
}

/// The 4-bit form of [`pack`].
pub fn pack_q4(data: &[u16], n: usize, k: usize) -> Vec<u8> {
    let blocks = n.div_ceil(16);
    let groups = k / GROUP;
    let block = groups * Q4_GROUP_BYTES;
    let mut out = vec![0u8; blocks * block];
    let f = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    for b in 0..blocks {
        for g in 0..groups {
            let base = b * block + g * Q4_GROUP_BYTES;
            for lane in 0..16 {
                let row = b * 16 + lane;
                if row >= n {
                    continue;
                }
                let w = &data[row * k + g * GROUP..row * k + (g + 1) * GROUP];
                let grid8 = crate::jit::knobs::knobs().neon.q4_grid8;
                // `absmax / 7` (codes in ±7), or the `Q4_0` rule: `signed_max / -8`, the value of
                // largest magnitude on −8.
                let signed_max =
                    w.iter()
                        .map(|&x| f(x))
                        .fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
                let (scale, inv) = match (signed_max != 0.0, grid8) {
                    (false, _) => (0.0, 0.0),
                    (true, false) => (signed_max.abs() / 7.0, 7.0 / signed_max.abs()),
                    (true, true) => (signed_max / -8.0, -8.0 / signed_max),
                };
                for (i, &x) in w.iter().enumerate() {
                    let signed = (f(x) * inv).round_ties_even().clamp(-8.0, 7.0) as i8;
                    // Two's complement for `sdot` (AArch64), offset binary `q + 8` for `vpdpbusd`
                    // (x86-64).
                    let code = if X86 {
                        (signed + 8) as u8
                    } else {
                        signed as u8 & 0xf
                    };
                    // step t = i / 8, byte j = i % 4, high nibble when (i % 8) >= 4.
                    let at = base + (i / 8) * 64 + (lane / 4) * 16 + (lane % 4) * 4 + i % 4;
                    out[at] |= if i % 8 >= 4 { code << 4 } else { code };
                }
                out[base + 256 + lane * 4..base + 256 + lane * 4 + 4]
                    .copy_from_slice(&(if X86 { scale } else { scale / 16.0 }).to_le_bytes());
            }
        }
    }
    out
}

/// The int8 form of [`pack`].
pub fn pack_int8(data: &[u16], n: usize, k: usize) -> Vec<u8> {
    let blocks = n.div_ceil(16);
    let groups = k / GROUP;
    let mut out = vec![0u8; blocks * groups * GROUP_BYTES];
    let f = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    for block in 0..blocks {
        for g in 0..groups {
            let base = block * groups * GROUP_BYTES + g * GROUP_BYTES;
            for lane in 0..16 {
                let row = block * 16 + lane;
                if row >= n {
                    continue;
                }
                let w = &data[row * k + g * GROUP..row * k + (g + 1) * GROUP];
                let max = w.iter().map(|&b| f(b).abs()).fold(0.0f32, f32::max);
                let (scale, inv) = if max > 0.0 {
                    (max / 127.0, 127.0 / max)
                } else {
                    (0.0, 0.0)
                };
                for (i, &b) in w.iter().enumerate() {
                    let q = (f(b) * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                    // step = i / 4, quad = lane / 4, row in quad = lane % 4, byte in step = i % 4.
                    out[base + (i / 4) * 64 + (lane / 4) * 16 + (lane % 4) * 4 + i % 4] =
                        if X86 { (q as u8) ^ 0x80 } else { q as u8 };
                }
                out[base + 512 + lane * 4..base + 512 + lane * 4 + 4]
                    .copy_from_slice(&scale.to_le_bytes());
            }
        }
    }
    out
}

/// Tries of [`q4_weighted_scale`]: nineteen starting scales, two refits each.
pub const Q4_WEIGHTED_TRIES: usize = 38;

/// The Σ of 32 terms, by eight partial sums (the lanes `j % 8`) then a pairwise reduction: a fixed
/// order, which the compiler keeps in vectors.
#[inline(always)]
fn sum32(f: impl Fn(usize) -> f32) -> f32 {
    let mut s = [0.0f32; 8];
    for j in 0..GROUP {
        s[j % 8] += f(j);
    }
    ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]))
}

/// `rne(w · inv)` clamped to `[−8, 7]`. The rounding by `(x + 1.5·2²³) − 1.5·2²³` is
/// `round_ties_even` for `|x| < 2²²` (but for the sign of a zero) and lands past the clamp beyond:
/// the same codes, in vectors, where the x86-64 baseline (SSE2, no `roundps`) would call
/// `roundevenf` per value.
#[inline(always)]
fn q4_codes(w: &[f32; GROUP], inv: f32) -> [f32; GROUP] {
    const MAGIC: f32 = 12_582_912.0;
    std::array::from_fn(|j| ((w[j] * inv + MAGIC) - MAGIC).clamp(-8.0, 7.0))
}

/// The least-squares scale of codes `q` under the weights `imp`: `Σ imp·w·q / Σ imp·q²` (0 when the
/// denominator is).
#[inline(always)]
fn q4_refit(w: &[f32; GROUP], imp: &[f32; GROUP], q: &[f32; GROUP]) -> f32 {
    let sq = sum32(|j| imp[j] * q[j] * q[j]);
    if sq > 0.0 {
        sum32(|j| imp[j] * w[j] * q[j]) / sq.max(1e-30)
    } else {
        0.0
    }
}

/// The `Q4_0` scale and codes of 32 weights `w` that minimize `Σ imp_k · (w_k − d·q_k)²`, by the
/// search of the quantization simulator of the 2026-11-12 measurement (`q4_search`, `wmode` `imp`,
/// refit on): for `s` in −9..=9, codes `q = rne(w · (1/m) · −(8 + 0.1 s))` clamped to `[−8, 7]`
/// (`m` the value of largest magnitude, signed, the first one on a tie), their least-squares scale
/// `d`, then `q2 = rne(w / d)` and its scale `d2`; of the 38 pairs the first of smallest weighted
/// error (strictly smaller to replace). Also returns which try won (`2·(s + 9) + refit`, refit 0 or
/// 1). Same storage as the grid: one f32 scale, codes in `[−8, 7]`.
pub fn q4_weighted_scale(w: &[f32; GROUP], imp: &[f32; GROUP]) -> (f32, [f32; GROUP], usize) {
    let m = w
        .iter()
        .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
    // torch's `den / m` is `reciprocal(m) · den`.
    let rm = if m != 0.0 { 1.0 / m } else { 0.0 };
    let err = |q: &[f32; GROUP], d: f32| {
        sum32(|j| {
            let e = w[j] - q[j] * d;
            imp[j] * e * e
        })
    };
    let (mut best_e, mut best_d, mut best_q, mut won) =
        (f32::INFINITY, 0.0f32, [0.0f32; GROUP], 0usize);
    for si in 0..19 {
        let den = -(8.0f64 + 0.1 * (si as f64 - 9.0)) as f32;
        let q = q4_codes(w, rm * den);
        let d = q4_refit(w, imp, &q);
        let q2 = q4_codes(w, if d != 0.0 { 1.0 / d } else { 0.0 });
        let d2 = q4_refit(w, imp, &q2);
        for (c, (qq, dd)) in [(q, d), (q2, d2)].into_iter().enumerate() {
            let e = err(&qq, dd);
            if e < best_e {
                (best_e, best_d, best_q, won) = (e, dd, qq, 2 * si + c);
            }
        }
    }
    (best_d, best_q, won)
}

/// [`pack_q4`]'s layout with the scale of each group of each row chosen by [`q4_weighted_scale`]
/// under `imp`, the importance of each input column (`k` values: `E[x²]` of the projection's input
/// on calibration texts, `engine::xlmr`), instead of the grid. The kernels read it as they read
/// [`pack_q4`]'s. The R16 blocks are spread over `workers` threads of the global pool (1: the
/// calling thread). Also returns how often each of the [`Q4_WEIGHTED_TRIES`] won.
pub fn pack_q4_weighted(
    data: &[u16],
    n: usize,
    k: usize,
    imp: &[f32],
    workers: usize,
) -> (Vec<u8>, [u64; Q4_WEIGHTED_TRIES]) {
    assert!(k.is_multiple_of(GROUP) && imp.len() == k && data.len() == n * k);
    let blocks = n.div_ceil(16);
    let groups = k / GROUP;
    let block = groups * Q4_GROUP_BYTES;
    let mut out = vec![0u8; blocks * block];
    let f = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let one = |b: usize, dst: &mut [u8], hist: &mut [u64; Q4_WEIGHTED_TRIES]| {
        for g in 0..groups {
            let base = g * Q4_GROUP_BYTES;
            let ig: &[f32; GROUP] = imp[g * GROUP..(g + 1) * GROUP].try_into().unwrap();
            for lane in 0..16 {
                let row = b * 16 + lane;
                if row >= n {
                    continue;
                }
                let w: [f32; GROUP] = std::array::from_fn(|i| f(data[row * k + g * GROUP + i]));
                let (scale, q, won) = q4_weighted_scale(&w, ig);
                hist[won] += 1;
                for (i, &c) in q.iter().enumerate() {
                    let signed = c as i8;
                    // As `pack_q4`: two's complement (AArch64), `q + 8` (x86-64).
                    let code = if X86 {
                        (signed + 8) as u8
                    } else {
                        signed as u8 & 0xf
                    };
                    let at = base + (i / 8) * 64 + (lane / 4) * 16 + (lane % 4) * 4 + i % 4;
                    dst[at] |= if i % 8 >= 4 { code << 4 } else { code };
                }
                dst[base + 256 + lane * 4..base + 256 + lane * 4 + 4]
                    .copy_from_slice(&(if X86 { scale } else { scale / 16.0 }).to_le_bytes());
            }
        }
    };
    let mut total = [0u64; Q4_WEIGHTED_TRIES];
    if workers <= 1 || blocks <= 1 {
        for (b, dst) in out.chunks_mut(block).enumerate() {
            one(b, dst, &mut total);
        }
        return (out, total);
    }
    {
        // One R16 block per claim: disjoint byte ranges, each locked once.
        let slots: Vec<std::sync::Mutex<&mut [u8]>> =
            out.chunks_mut(block).map(std::sync::Mutex::new).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let sums = std::sync::Mutex::new([0u64; Q4_WEIGHTED_TRIES]);
        let run = |_: usize| {
            let mut hist = [0u64; Q4_WEIGHTED_TRIES];
            loop {
                let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(slot) = slots.get(b) else { break };
                one(b, &mut slot.lock().unwrap(), &mut hist);
            }
            let mut s = sums.lock().unwrap();
            for (t, h) in s.iter_mut().zip(hist) {
                *t += h;
            }
        };
        let pool = crate::pool::thread_pool::global_pool();
        if pool
            .collective(workers.min(pool.num_workers()).max(1), run, || {})
            .is_err()
        {
            // No collective (inside one already): the calling thread does what is left.
            run(0);
        }
        total = *sums.lock().unwrap();
    }
    (out, total)
}

#[cfg(test)]
mod weighted_tests {
    use super::*;

    /// The values a layout of [`pack_q4`] holds: `[n][k]` codes and `[n][k/32]` scales (the stored
    /// f32, ×16 on AArch64).
    fn unpack(bytes: &[u8], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
        let groups = k / GROUP;
        let (mut q, mut s) = (vec![0i8; n * k], vec![0f32; n * groups]);
        for row in 0..n {
            let (b, lane) = (row / 16, row % 16);
            for g in 0..groups {
                let base = (b * groups + g) * Q4_GROUP_BYTES;
                for i in 0..GROUP {
                    let byte =
                        bytes[base + (i / 8) * 64 + (lane / 4) * 16 + (lane % 4) * 4 + i % 4];
                    let nib = if i % 8 >= 4 { byte >> 4 } else { byte & 0xf };
                    q[row * k + g * GROUP + i] = if X86 {
                        nib as i8 - 8
                    } else {
                        ((nib << 4) as i8) >> 4
                    };
                }
                let at = base + 256 + lane * 4;
                let v = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                s[row * groups + g] = if X86 { v } else { v * 16.0 };
            }
        }
        (q, s)
    }

    fn weights(n: usize, k: usize) -> Vec<u16> {
        let mut x = 0x9e3779b97f4a7c15u64;
        (0..n * k)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let u = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                // A few outliers, as in trained weights.
                let v = if i % 97 == 0 { u * 8.0 } else { u };
                crate::engine::tensors::to_bf16(v * 0.05)
            })
            .collect()
    }

    fn bf(b: u16) -> f32 {
        f32::from_bits(u32::from(b) << 16)
    }

    /// Σ imp·(w − d·q)² of each group, in f64.
    fn werr(data: &[u16], q: &[i8], s: &[f32], imp: &[f32], k: usize) -> Vec<f64> {
        (0..data.len() / GROUP)
            .map(|rg| {
                (0..GROUP)
                    .map(|i| {
                        let at = rg * GROUP + i;
                        let e = bf(data[at]) as f64 - q[at] as f64 * s[rg] as f64;
                        imp[(rg % (k / GROUP)) * GROUP + i] as f64 * e * e
                    })
                    .sum()
            })
            .collect()
    }

    /// Valid layout (codes in [−8, 7], one try counted per group); the weighted error of every
    /// group no larger than the grid's (`Q4_0`, `signed_max / −8`, whose codes the search
    /// refits); the importance scaled by a power of two gives the same bytes; the pool gives
    /// the bytes of one thread; importance 1 gives the unweighted search (the same tries, error
    /// Σ (w − d·q)², recomputed in f64).
    #[test]
    fn weighted_q4_packer() {
        let (n, k) = (40, 96);
        let data = weights(n, k);
        let mut x = 7u64;
        let imp: Vec<f32> = (0..k)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / (1u64 << 31) as f32).powi(3) * 4.0
                    + if x.is_multiple_of(5) { 0.0 } else { 1e-3 }
            })
            .collect();
        // The rounding of `q4_codes` is `round_ties_even` (ties, both signs, large values,
        // infinities).
        let xs: Vec<f32> = (-40_000..40_000)
            .map(|i| i as f32 / 4096.0)
            .chain([
                -0.5,
                0.5,
                1.5,
                2.5,
                -2.5,
                7.5,
                -8.5,
                4.0e6,
                -4.0e6,
                1.0e30,
                -1.0e30,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ])
            .collect();
        for c in xs.chunks(GROUP) {
            let mut w = [0.0f32; GROUP];
            w[..c.len()].copy_from_slice(c);
            let q = q4_codes(&w, 1.0);
            for j in 0..GROUP {
                assert_eq!(q[j], w[j].round_ties_even().clamp(-8.0, 7.0), "{}", w[j]);
            }
        }
        let (bytes, hist) = pack_q4_weighted(&data, n, k, &imp, 1);
        assert_eq!(bytes.len(), n.div_ceil(16) * block_bytes(k, true));
        assert_eq!(hist.iter().sum::<u64>(), (n * k / GROUP) as u64);
        let (q, s) = unpack(&bytes, n, k);
        assert!(q.iter().all(|&c| (-8..=7).contains(&c)) && s.iter().all(|v| v.is_finite()));
        // The grid of `pack_q4` under `q4_grid8`, recomputed (`pack_q4` reads the knob).
        let (mut gq, mut gs) = (vec![0i8; n * k], vec![0f32; n * k / GROUP]);
        for rg in 0..n * k / GROUP {
            let w = &data[rg * GROUP..(rg + 1) * GROUP];
            let m = w
                .iter()
                .map(|&x| bf(x))
                .fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
            let inv = if m != 0.0 { -8.0 / m } else { 0.0 };
            gs[rg] = m / -8.0;
            for i in 0..GROUP {
                gq[rg * GROUP + i] = (bf(w[i]) * inv).round_ties_even().clamp(-8.0, 7.0) as i8;
            }
        }
        let (e, ge) = (werr(&data, &q, &s, &imp, k), werr(&data, &gq, &gs, &imp, k));
        for (a, b) in e.iter().zip(&ge) {
            assert!(*a <= b * (1.0 + 1e-5) + 1e-12, "weighted {a} > grid {b}");
        }
        assert!(e.iter().sum::<f64>() < 0.9 * ge.iter().sum::<f64>());
        let imp4: Vec<f32> = imp.iter().map(|v| v * 4.0).collect();
        assert_eq!(pack_q4_weighted(&data, n, k, &imp4, 1).0, bytes);
        assert_eq!(pack_q4_weighted(&data, n, k, &imp, 4).0, bytes);
        let ones = vec![1.0f32; k];
        let (ub, _) = pack_q4_weighted(&data, n, k, &ones, 1);
        let (uq, us) = unpack(&ub, n, k);
        let ue = werr(&data, &uq, &us, &ones, k);
        for rg in 0..n * k / GROUP {
            let w: [f32; GROUP] = std::array::from_fn(|i| bf(data[rg * GROUP + i]));
            let m = w
                .iter()
                .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let mut best = f64::INFINITY;
            for si in 0..19 {
                let inv = (1.0 / m) * (-(8.0f64 + 0.1 * (si as f64 - 9.0)) as f32);
                let mut q: Vec<f64> = w
                    .iter()
                    .map(|&v| (v * inv).round_ties_even().clamp(-8.0, 7.0) as f64)
                    .collect();
                for _ in 0..2 {
                    let d = w.iter().zip(&q).map(|(&v, &c)| v as f64 * c).sum::<f64>()
                        / q.iter().map(|c| c * c).sum::<f64>();
                    best = best.min(
                        w.iter()
                            .zip(&q)
                            .map(|(&v, &c)| (v as f64 - c * d).powi(2))
                            .sum(),
                    );
                    q = w
                        .iter()
                        .map(|&v| (v * (1.0 / d as f32)).round_ties_even().clamp(-8.0, 7.0) as f64)
                        .collect();
                }
            }
            assert!(
                (ue[rg] - best).abs() <= 1e-4 * best + 1e-14,
                "group {rg}: {} vs {best}",
                ue[rg]
            );
        }
    }
}
