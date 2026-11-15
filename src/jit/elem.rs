//! The driver of the elementwise kernels: compilation and cache.
//!
//! One blob per `(operation, width)`, no plan: an RMSNorm over 2560 values or
//! a SwiGLU over 9216 costs a few microseconds and lives in L1. The ISA of the
//! arm — zmm on AVX-512, ymm on AVX2 — is chosen once per process from what
//! the machine has, and the two arms agree to one bf16 ulp, not to the bit
//! (the gap is measured).

use crate::jit::emit_elem::{emit_elem, ElemSpec};

/// Which ISA the elementary generator uses on **this** machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElemIsa {
    /// AVX-512(+BF16) : `vdpbf16ps`, `vcvtneps2bf16`, tiles zmm.
    Zmm,
    /// AVX2 + FMA : tiles ymm, `vdpbf16ps` and `vcvtneps2bf16` **emulated**.
    Ymm,
    /// AArch64 NEON with BF16: widened by `shll`, narrowed by `bfcvtn`.
    Neon,
    /// RISC-V RVV: widened by `vfwcvtbf16`, narrowed by `vfncvtbf16`.
    Rvv,
}

/// Chooses the emission ISA **based on runtime detection**, and nothing else.
///
/// No setting enters here: the JIT emits for the machine it runs on. AVX-512
/// first, because its tiles have 16 lanes instead of 8 and `vcvtneps2bf16`
/// costs one instruction instead of five.
///
/// The `Ymm` arm returns the **same result to one bf16 ulp**, not the same
/// bits. Bit-for-bit identity holds **at a fixed ISA**, not from one arm to
/// the other, for three distinct reasons, all measured and all checked:
///
/// 1. **The summation order of Σx²** (`RmsNorm` only): the two arms do not partition the same
///    elements into the same accumulator lanes — sixteen against eight — and do not reduce
///    horizontally by the same tree. Floating addition is not associative: Σx² differs on 45 % of
///    the draws, and the bf16 output on 3.7 coefficients per million at `dim = 2048`.
/// 2. **The denormals** (every operation): `vcvtneps2bf16` returns `±0` as soon as its f32 input
///    has a zero exponent field, where the software rounding of the `Ymm` arm, purely integer,
///    keeps them.
/// 3. **The NaNs**: the software rounding does not always return a `NaN`; depending on the payload
///    it can give an infinity or a zero (see `arch/x86/emit_elem_avx2.rs`).
///
/// This is not a degraded mode: on normal values and outside `RmsNorm` it is
/// the same computation, slower. It is not the same bits.
pub fn select_isa() -> Option<ElemIsa> {
    match crate::jit::engine_isa() {
        crate::jit::IsaEngine::Avx512 => Some(ElemIsa::Zmm),
        crate::jit::IsaEngine::Avx2 => Some(ElemIsa::Ymm),
        crate::jit::IsaEngine::NeonBf16 => Some(ElemIsa::Neon),
        crate::jit::IsaEngine::Rvv => Some(ElemIsa::Rvv),
        crate::jit::IsaEngine::BelowFloor => None,
    }
}

use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// A compiled elementary kernel.
pub struct CompiledElem {
    _exec: Mmap,
    entry: *const u8,
    code_bytes: usize,
}

// SAFETY: the pointer targets a shared, immutable RX page, whose ownership is held by `Mmap`;
// the generated code has no global state.
unsafe impl Send for CompiledElem {}
unsafe impl Sync for CompiledElem {}

impl CompiledElem {
    pub(crate) fn lowering_bytes(&self) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.entry, self.code_bytes).to_vec() }
    }
}

/// One blob on its own executable page.
fn map_blob(blob: &[u8]) -> Result<Mmap> {
    crate::jit::arch::map_exec(blob.len().max(64), |map| {
        map[..blob.len()].copy_from_slice(blob)
    })
    .map_err(|e| HerbertError::Backend(format!("executable mapping: {e}")))
}

type Cache = RwLock<HashMap<ElemSpec, Option<Arc<CompiledElem>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Compile `spec`, or return the already cached version.
///
/// `Ok(None)` = shape refused, the caller falls back. `Err` = generator
/// failure, which must propagate.
pub fn get_or_compile(spec: ElemSpec) -> Result<Option<Arc<CompiledElem>>> {
    if select_isa().is_none() {
        return Ok(None);
    }
    if spec.check().is_err() {
        return Ok(None);
    }
    if let Some(hit) = cache().read().unwrap().get(&spec) {
        return Ok(hit.clone());
    }
    // Shape refusals were handled by `spec.check()` above. Any error returned
    // by an emitter here is a generator failure and must reach the caller; it
    // must not be disguised as an unsupported shape.
    let blob = emit_elem(&spec)?;
    let exec = map_blob(&blob)?;
    crate::jit::perf_map::register(exec.as_ptr(), blob.len(), || {
        format!("elem_{}", format!("{spec:?}").replace(' ', ""))
    });
    let built = Some(Arc::new(CompiledElem {
        entry: exec.as_ptr(),
        code_bytes: blob.len(),
        _exec: exec,
    }));
    if crate::jit::options::debug_plan() {
        eprintln!(
            "DEBUG [backend_jit] elem {} dim={} bytes={}",
            spec.op.name(),
            spec.dim,
            blob.len()
        );
    }
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(spec).or_insert(built).clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::emit_elem::{
        ClampFn, DepthwiseRowsFn, ElemOp, LocalAttentionBlockFn, LocalAttentionCall,
        LocalAttentionFn, PoolRowsFn, RmsNormFn, SwiGluFn,
    };

    fn bf16(x: f32) -> u16 {
        crate::engine::packs::to_bf16(x)
    }
    fn f32_of(b: u16) -> f32 {
        f32::from_bits(u32::from(b) << 16)
    }
    /// bf16 values in about [-4, 4), from a linear congruential generator.
    fn values(n: usize, seed: u32) -> Vec<u16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bf16(((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 8.0)
            })
            .collect()
    }
    /// The blobs of `spec` this machine can run: the arm of the process, and on x86-64 with AVX-512
    /// the AVX2 one too.
    fn arms(spec: ElemSpec) -> Vec<(&'static str, Mmap)> {
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut out = vec![("process", map_blob(&emit_elem(&spec).unwrap()).unwrap())];
        #[cfg(target_arch = "x86_64")]
        if crate::core::isa::feature("avx2")
            && crate::core::isa::feature("fma")
            && select_isa() == Some(ElemIsa::Zmm)
        {
            out.push((
                "avx2",
                map_blob(&crate::jit::emit_elem_avx2::emit_elem_avx2(&spec).unwrap()).unwrap(),
            ));
        }
        out
    }

    /// `Clamp` gives `min(max(x, lo), hi)` bit for bit, in place too.
    #[test]
    fn clamp_bounds_every_value() {
        if select_isa().is_none() {
            return;
        }
        let dim = 96;
        let x = values(dim, 7);
        let (lo, hi) = (f32_of(bf16(-1.25)), f32_of(bf16(2.5)));
        let want: Vec<u16> = x.iter().map(|&v| bf16(f32_of(v).max(lo).min(hi))).collect();
        for (arm, code) in arms(ElemSpec {
            op: ElemOp::Clamp,
            dim,
        }) {
            let f: ClampFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let mut y = vec![0u16; dim];
            unsafe { f(x.as_ptr(), y.as_mut_ptr(), lo, hi) };
            assert_eq!(y, want, "{arm}");
            let mut z = x.clone();
            unsafe { f(z.as_ptr(), z.as_mut_ptr(), lo, hi) };
            assert_eq!(z, want, "{arm} in place");
        }
    }

    /// `PoolRows` gives `bf16(bf16(Σ x[r]·inv) · s)`, the sums fused in row order from zero, bit
    /// for bit.
    #[test]
    fn pool_rows_is_the_mean_scaled() {
        if select_isa().is_none() {
            return;
        }
        let (dim, rows) = (128, 9);
        let x = values(dim * rows, 11);
        let (s, inv) = (f32_of(bf16(768f32.sqrt())), 1.0 / 9.0f32);
        let want: Vec<u16> = (0..dim)
            .map(|i| {
                let acc = (0..rows).fold(0f32, |a, r| f32_of(x[r * dim + i]).mul_add(inv, a));
                bf16(f32_of(bf16(acc)) * s)
            })
            .collect();
        for (arm, code) in arms(ElemSpec {
            op: ElemOp::PoolRows(rows as u8),
            dim,
        }) {
            let f: PoolRowsFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let mut y = vec![0u16; dim];
            unsafe { f(x.as_ptr(), y.as_mut_ptr(), s, inv) };
            assert_eq!(y, want, "{arm}");
        }
    }

    /// Distance in bf16 steps between two bf16 values of the same sign (or both near zero).
    fn ulps(a: u16, b: u16) -> u32 {
        let key = |x: u16| {
            if x & 0x8000 != 0 {
                0x8000i32 - i32::from(x & 0x7fff)
            } else {
                0x8000 + i32::from(x)
            }
        };
        key(a).abs_diff(key(b))
    }

    /// `LayerNorm` against the reference in f64 (two-pass mean and biased variance): within one
    /// bf16 step.
    #[test]
    fn layer_norm_is_torch_layer_norm() {
        if select_isa().is_none() {
            return;
        }
        // 3 840 and 6 912: the 12B's embedder, the offset of `b` past an AArch64 `add` immediate.
        for dim in [64usize, 1024, 3840, 6912] {
            let x: Vec<u16> = values(dim, 5)
                .iter()
                .map(|&v| bf16(f32_of(v) + 3.0))
                .collect();
            let mut w = values(dim, 9);
            w.extend(values(dim, 13));
            let eps = 1e-5f32;
            let xf: Vec<f64> = x.iter().map(|&v| f32_of(v) as f64).collect();
            let mu = xf.iter().sum::<f64>() / dim as f64;
            let var = xf.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / dim as f64;
            let inv = 1.0 / (var + eps as f64).sqrt();
            let want: Vec<u16> = (0..dim)
                .map(|i| {
                    bf16(
                        ((xf[i] - mu) * inv * f32_of(w[i]) as f64 + f32_of(w[dim + i]) as f64)
                            as f32,
                    )
                })
                .collect();
            for (arm, code) in arms(ElemSpec {
                op: ElemOp::LayerNorm,
                dim,
            }) {
                let f: RmsNormFn = unsafe { std::mem::transmute(code.as_ptr()) };
                let mut y = vec![0u16; dim];
                unsafe { f(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), eps) };
                let worst = y
                    .iter()
                    .zip(&want)
                    .map(|(&a, &b)| ulps(a, b))
                    .max()
                    .unwrap();
                assert!(worst <= 1, "{arm} dim {dim}: {worst} bf16 steps");
            }
        }
    }

    /// `GeluErf` against `g · ½·(1 + erf(g/√2)) · up` in f64 (`libm`-free: erf by its series and
    /// continued fraction through `erfc` of f64 precision): within one bf16 step, the widths
    /// unrolled and looped.
    #[test]
    fn gelu_erf_is_the_exact_gelu() {
        if select_isa().is_none() {
            return;
        }
        // erf in f64 by the Taylor series (|x| < 3) or 1 − erfc by its continued fraction.
        fn erf(x: f64) -> f64 {
            let a = x.abs();
            let r = if a < 3.0 {
                let (mut term, mut sum, mut n) = (a, a, 0.0f64);
                loop {
                    n += 1.0;
                    term *= -a * a / n;
                    let add = term / (2.0 * n + 1.0);
                    sum += add;
                    if add.abs() < 1e-17 {
                        break;
                    }
                }
                sum * 2.0 / std::f64::consts::PI.sqrt()
            } else {
                let mut f = 0.0f64;
                for k in (1..60).rev() {
                    f = (k as f64 / 2.0) / (a + f);
                }
                1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f)
            };
            r.copysign(x)
        }
        for dim in [96usize, 4096] {
            let g: Vec<u16> = values(dim, 21)
                .iter()
                .map(|&v| bf16(f32_of(v) * 1.5))
                .collect();
            let u = values(dim, 23);
            let want: Vec<u16> = (0..dim)
                .map(|i| {
                    let x = f32_of(g[i]) as f64;
                    bf16(
                        (x * 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2)) * f32_of(u[i]) as f64)
                            as f32,
                    )
                })
                .collect();
            for (arm, code) in arms(ElemSpec {
                op: ElemOp::GeluErf,
                dim,
            }) {
                let f: SwiGluFn = unsafe { std::mem::transmute(code.as_ptr()) };
                let mut y = vec![0u16; dim];
                unsafe { f(g.as_ptr(), u.as_ptr(), y.as_mut_ptr()) };
                let (at, worst) = y
                    .iter()
                    .zip(&want)
                    .map(|(&a, &b)| ulps(a, b))
                    .enumerate()
                    .max_by_key(|e| e.1)
                    .unwrap();
                assert!(
                    worst <= 1,
                    "{arm} dim {dim}: {worst} bf16 steps at {at}: gelu({}) · {} gives {}, not {}",
                    f32_of(g[at]),
                    f32_of(u[at]),
                    f32_of(y[at]),
                    f32_of(want[at])
                );
            }
        }
    }

    /// `DepthwiseRows` gives `bf16(Σᵣ x[r]·w[r])`, the products fused in row order from zero, bit
    /// for bit.
    #[test]
    fn depthwise_rows_is_the_causal_convolution() {
        if select_isa().is_none() {
            return;
        }
        let (dim, taps) = (1024, 5);
        let x = values(dim * taps, 13);
        let w = values(dim * taps, 17);
        let want: Vec<u16> = (0..dim)
            .map(|i| {
                bf16((0..taps).fold(0f32, |a, r| {
                    f32_of(x[r * dim + i]).mul_add(f32_of(w[r * dim + i]), a)
                }))
            })
            .collect();
        for (arm, code) in arms(ElemSpec {
            op: ElemOp::DepthwiseRows(taps as u8),
            dim,
        }) {
            let f: DepthwiseRowsFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let mut y = vec![0u16; dim];
            unsafe { f(x.as_ptr(), w.as_ptr(), y.as_mut_ptr()) };
            assert_eq!(y, want, "{arm}");
        }
    }

    /// `LocalAttention` against an f64 reference, at every row of the window's start (the masked
    /// keys) and past it: within two bf16 steps of the largest output.
    #[test]
    fn local_attention_is_the_audio_tower_attention() {
        if select_isa().is_none() {
            return;
        }
        let (heads, hd, window) = (8usize, 128usize, 12usize);
        let width = heads * hd;
        let rows = 20;
        let pad = window - 1;
        // k, v with `pad` zero rows first; small values, as the tower's.
        let mut k = vec![0u16; (pad + rows) * width];
        let mut v = vec![0u16; (pad + rows) * width];
        k[pad * width..].copy_from_slice(&values(rows * width, 3));
        v[pad * width..].copy_from_slice(&values(rows * width, 5));
        let q = values(rows * width, 7);
        let sp: Vec<u16> = values(hd, 9)
            .iter()
            .map(|&b| bf16(f32_of(b).abs() * 0.25 + 0.5))
            .collect();
        let relk = values((window + 1) * width, 11);
        let mut rel = sp.clone();
        rel.extend_from_slice(&relk);
        let (qs, ks, cap) = (
            (hd as f64).powf(-0.5) / std::f64::consts::LN_2,
            (1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2,
            50.0f64,
        );
        let spec = ElemSpec {
            op: ElemOp::LocalAttention {
                heads: heads as u8,
                window: window as u8,
                rows: 0,
            },
            dim: hd,
        };
        let code = map_blob(&emit_elem(&spec).unwrap()).unwrap();
        let f: LocalAttentionFn = unsafe { std::mem::transmute(code.as_ptr()) };
        let g = |b: u16| f64::from(f32_of(b));
        for t in 0..rows {
            let mut y = vec![0u16; width];
            unsafe {
                f(
                    q[t * width..].as_ptr(),
                    k[(pad + t) * width..].as_ptr(),
                    v[(pad + t) * width..].as_ptr(),
                    rel.as_ptr(),
                    y.as_mut_ptr(),
                    t,
                    qs as f32,
                    ks as f32,
                    cap as f32,
                )
            };
            for h in 0..heads {
                let o = h * hd;
                let qq: Vec<f64> = (0..hd)
                    .map(|d| g(q[t * width + o + d]) * qs * g(sp[d]))
                    .collect();
                let n = (t + 1).min(window);
                let l: Vec<f64> = (0..n)
                    .map(|a| {
                        let kr = (pad + t - a) * width + o;
                        let rr = (window - a) * width + o;
                        let ac: f64 = (0..hd).map(|d| qq[d] * g(k[kr + d])).sum();
                        let bd: f64 = (0..hd).map(|d| qq[d] * g(relk[rr + d])).sum();
                        cap * ((ac * ks + bd) / cap).tanh()
                    })
                    .collect();
                let m = l.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = l.iter().map(|x| (x - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                let want: Vec<f64> = (0..hd)
                    .map(|d| {
                        (0..n)
                            .map(|a| e[a] / sum * g(v[(pad + t - a) * width + o + d]))
                            .sum()
                    })
                    .collect();
                let big = want.iter().fold(0f64, |a, &x| a.max(x.abs()));
                for d in 0..hd {
                    let got = g(y[o + d]);
                    assert!(
                        (got - want[d]).abs() <= big * 2.0 / 128.0 + 1e-6,
                        "row {t} head {h} lane {d}: {got} for {}",
                        want[d]
                    );
                }
            }
        }
    }

    /// The block kernels of `spec` this machine runs: the process's arm, and on x86-64 with AVX-512
    /// the AVX2 one too.
    fn block_arms(spec: ElemSpec) -> Vec<(&'static str, Mmap)> {
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut out = vec![(
            "process",
            map_blob(&emit_elem(&spec).unwrap_or_else(|e| panic!("{spec:?} refused: {e}")))
                .unwrap(),
        )];
        #[cfg(target_arch = "x86_64")]
        if select_isa() == Some(ElemIsa::Zmm) {
            out.push((
                "avx2",
                map_blob(
                    &crate::jit::arch::x86::emit_local_attn::emit_block_arm(&spec, false).unwrap(),
                )
                .unwrap(),
            ));
        }
        out
    }

    /// The inputs of a local attention of `heads × hd`, `window`, for `rows` rows padded to
    /// `padded`: q, k and v with `window − 1` zero rows first (the rows past `rows` other
    /// values, never to be seen), `sp` then the relative keys.
    fn attention_inputs(
        heads: usize,
        hd: usize,
        window: usize,
        rows: usize,
        padded: usize,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let width = heads * hd;
        let pad = window - 1;
        let small = |v: Vec<u16>| {
            v.into_iter()
                .map(|b| bf16(f32_of(b) * 0.125))
                .collect::<Vec<u16>>()
        };
        let mut k = vec![0u16; (pad + padded) * width];
        let mut v = vec![0u16; (pad + padded) * width];
        k[pad * width..].copy_from_slice(&small(values(padded * width, 3 + rows as u32)));
        v[pad * width..].copy_from_slice(&values(padded * width, 5 + rows as u32));
        let q = small(values(padded * width, 7 + rows as u32));
        let mut rel: Vec<u16> = values(hd, 9)
            .iter()
            .map(|&b| bf16(f32_of(b).abs() * 0.25 + 0.5))
            .collect();
        rel.extend_from_slice(&small(values((window + 1) * width, 11)));
        (q, k, v, rel)
    }

    /// The block kernel of `LocalAttention` gives the row kernel's bits, on every tile size, for
    /// row counts below, at and past a tile and a register group (the masked keys of the first
    /// rows, a partial last tile, tile edges off the window's multiples), and writes no row
    /// past the last.
    #[test]
    fn local_attention_block_is_the_row_kernel() {
        if cfg!(not(any(target_arch = "x86_64", target_arch = "aarch64"))) || select_isa().is_none()
        {
            return;
        }
        let (qs, ks, cap) = (
            (128f64).powf(-0.5) / std::f64::consts::LN_2,
            (1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2,
            50.0f64,
        );
        for (heads, hd, window) in [
            (8usize, 128usize, 12usize),
            (3, 64, 5),
            (2, 32, 16),
            (1, 96, 1),
        ] {
            let width = heads * hd;
            let pad = window - 1;
            let row_spec = ElemSpec {
                op: ElemOp::LocalAttention {
                    heads: heads as u8,
                    window: window as u8,
                    rows: 0,
                },
                dim: hd,
            };
            let row_code = map_blob(&emit_elem(&row_spec).unwrap()).unwrap();
            let row: LocalAttentionFn = unsafe { std::mem::transmute(row_code.as_ptr()) };
            for tile in [8usize, 16, 24, 32] {
                let spec = ElemSpec {
                    op: ElemOp::LocalAttention {
                        heads: heads as u8,
                        window: window as u8,
                        rows: tile as u8,
                    },
                    dim: hd,
                };
                for (arm, code) in block_arms(spec) {
                    let block: LocalAttentionBlockFn =
                        unsafe { std::mem::transmute(code.as_ptr()) };
                    for n in [
                        1usize, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 23, 24, 25, 31, 32,
                        33, 36, 47, 64, 100,
                    ] {
                        let padded = n.next_multiple_of(tile);
                        let (q, k, v, rel) = attention_inputs(heads, hd, window, n, padded);
                        let mut want = vec![0u16; n * width];
                        for t in 0..n {
                            unsafe {
                                row(
                                    q[t * width..].as_ptr(),
                                    k[(pad + t) * width..].as_ptr(),
                                    v[(pad + t) * width..].as_ptr(),
                                    rel.as_ptr(),
                                    want[t * width..].as_mut_ptr(),
                                    t,
                                    qs as f32,
                                    ks as f32,
                                    cap as f32,
                                )
                            };
                        }
                        const UNTOUCHED: u16 = 0x7FC1;
                        let mut y = vec![UNTOUCHED; padded * width];
                        let call = LocalAttentionCall {
                            q: q.as_ptr(),
                            k: k[pad * width..].as_ptr(),
                            v: v[pad * width..].as_ptr(),
                            rel: rel.as_ptr(),
                            y: y.as_mut_ptr(),
                            rows: &n,
                        };
                        for task in 0..n.div_ceil(tile) * heads {
                            unsafe { block(&call, task, qs as f32, ks as f32, cap as f32) };
                        }
                        let at = format!(
                            "{arm}: heads {heads} hd {hd} window {window}, tile {tile}, {n} rows"
                        );
                        for t in 0..n {
                            assert_eq!(
                                y[t * width..(t + 1) * width],
                                want[t * width..(t + 1) * width],
                                "{at}: row {t}"
                            );
                        }
                        assert!(
                            y[n * width..].iter().all(|&b| b == UNTOUCHED),
                            "{at}: a row past the last written"
                        );
                    }
                }
            }
        }
    }

    /// The row and block kernels of the audio tower's attention (E2B and E4B: 8 heads of 128,
    /// window 12) over 750 rows (30 s), one thread: `cargo test --release
    /// local_attention_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "a speed measurement, run by hand"]
    fn local_attention_timing() {
        let (heads, hd, window, n) = (8usize, 128usize, 12usize, 750usize);
        let width = heads * hd;
        let pad = window - 1;
        let (qs, ks, cap) = (0.1f32, 1.9f32, 50.0f32);
        let row_code = map_blob(
            &emit_elem(&ElemSpec {
                op: ElemOp::LocalAttention {
                    heads: 8,
                    window: 12,
                    rows: 0,
                },
                dim: hd,
            })
            .unwrap(),
        )
        .unwrap();
        let row: LocalAttentionFn = unsafe { std::mem::transmute(row_code.as_ptr()) };
        let reps = 50;
        let best = |f: &mut dyn FnMut()| {
            (0..reps)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::MAX, f64::min)
        };
        let (q, k, v, rel) = attention_inputs(heads, hd, window, n, n.next_multiple_of(32));
        let mut y = vec![0u16; n.next_multiple_of(32) * width];
        let t_row = best(&mut || {
            for t in 0..n {
                unsafe {
                    row(
                        q[t * width..].as_ptr(),
                        k[(pad + t) * width..].as_ptr(),
                        v[(pad + t) * width..].as_ptr(),
                        rel.as_ptr(),
                        y[t * width..].as_mut_ptr(),
                        t,
                        qs,
                        ks,
                        cap,
                    )
                };
            }
        });
        println!(
            "local attention, {n} rows, 8 × 128, window 12, one thread: row kernel {:.1} µs",
            t_row * 1e6
        );
        for (arm, code) in [8usize, 16, 24, 32].into_iter().flat_map(|tile| {
            block_arms(ElemSpec {
                op: ElemOp::LocalAttention {
                    heads: 8,
                    window: 12,
                    rows: tile as u8,
                },
                dim: hd,
            })
            .into_iter()
            .map(move |(arm, code)| ((arm, tile), code))
        }) {
            let (arm, tile) = arm;
            let block: LocalAttentionBlockFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let call = LocalAttentionCall {
                q: q.as_ptr(),
                k: k[pad * width..].as_ptr(),
                v: v[pad * width..].as_ptr(),
                rel: rel.as_ptr(),
                y: y.as_mut_ptr(),
                rows: &n,
            };
            let t_block = best(&mut || {
                for task in 0..n.div_ceil(tile) * heads {
                    unsafe { block(&call, task, qs, ks, cap) };
                }
            });
            println!(
                "  block kernel ({arm}), tiles of {tile}: {:.1} µs (× {:.2})",
                t_block * 1e6,
                t_row / t_block
            );
        }
    }

    /// The process arm's blob of `spec` when it serves a partial last block (AVX-512), `None`
    /// elsewhere.
    fn tailed(spec: ElemSpec) -> Option<Mmap> {
        if select_isa() != Some(ElemIsa::Zmm) {
            return None;
        }
        Some(map_blob(&emit_elem(&spec).unwrap()).unwrap())
    }

    /// The vision towers of 26B-A4B and Qwen3.5 (heads of 72): `RmsNorm` of 72 (and of 40, 1 160)
    /// within one bf16 step of f64; nothing written past the row.
    #[test]
    fn rms_norm_with_a_partial_block() {
        for dim in [72usize, 40, 1160] {
            let Some(code) = tailed(ElemSpec {
                op: ElemOp::RmsNorm,
                dim,
            }) else {
                return;
            };
            let f: RmsNormFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let x = values(dim, 21);
            let w = values(dim, 23);
            let eps = 1e-6f32;
            let ss: f64 = x.iter().map(|&v| (f32_of(v) as f64).powi(2)).sum();
            let inv = 1.0 / (ss / dim as f64 + eps as f64).sqrt();
            let mut y = vec![0x7fc1u16; dim + 40];
            unsafe { f(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), eps) };
            for i in 0..dim {
                let want = bf16((f32_of(x[i]) as f64 * inv * f32_of(w[i]) as f64) as f32);
                assert!(
                    ulps(y[i], want) <= 1,
                    "dim {dim} lane {i}: {:#x} for {want:#x}",
                    y[i]
                );
            }
            assert!(
                y[dim..].iter().all(|&v| v == 0x7fc1),
                "dim {dim}: written past the row"
            );
        }
    }

    /// `Rope` of 36 (18 pairs: a whole pass and a pass of 2) bit for bit against the scalar `a·c −
    /// b·s`, `a·s + b·c` (the product then the fused one, as the kernel); nothing written past
    /// the head.
    #[test]
    fn rope_with_a_partial_pass() {
        for dim in [36usize, 72, 20] {
            let Some(code) = tailed(ElemSpec {
                op: ElemOp::Rope,
                dim,
            }) else {
                return;
            };
            let f: crate::jit::emit_elem::RopeFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let half = dim / 2;
            let x0 = values(dim, 31);
            let cos: Vec<u16> = values(half, 33)
                .iter()
                .map(|&v| bf16((f32_of(v) / 4.0).cos()))
                .collect();
            let sin: Vec<u16> = values(half, 33)
                .iter()
                .map(|&v| bf16((f32_of(v) / 4.0).sin()))
                .collect();
            let mut x = x0.clone();
            x.extend(std::iter::repeat_n(0x7fc1u16, 40));
            unsafe { f(x.as_mut_ptr(), cos.as_ptr(), sin.as_ptr()) };
            for i in 0..half {
                let (a, b, c, s) = (
                    f32_of(x0[i]),
                    f32_of(x0[i + half]),
                    f32_of(cos[i]),
                    f32_of(sin[i]),
                );
                assert_eq!(x[i], bf16((-b).mul_add(s, a * c)), "dim {dim} lane {i}");
                assert_eq!(
                    x[i + half],
                    bf16(b.mul_add(c, a * s)),
                    "dim {dim} lane {}",
                    i + half
                );
            }
            assert!(
                x[dim..].iter().all(|&v| v == 0x7fc1),
                "dim {dim}: written past the head"
            );
        }
    }

    /// `GeluTanh` with a partial last block, unrolled (4 304: 134 blocks and one of 16) and looped
    /// (4 112: 128 blocks in 8 turns, then one of 16): bit for bit the first values of the
    /// whole-block kernel of 4 320 on the same rows; nothing written past the row.
    #[test]
    fn gelu_tanh_with_a_partial_block() {
        let whole = 4320usize;
        if tailed(ElemSpec {
            op: ElemOp::GeluTanh,
            dim: 64,
        })
        .is_none()
        {
            return;
        }
        let g = values(whole, 41);
        let u = values(whole, 43);
        let w = map_blob(
            &emit_elem(&ElemSpec {
                op: ElemOp::GeluTanh,
                dim: whole,
            })
            .unwrap(),
        )
        .unwrap();
        let fw: SwiGluFn = unsafe { std::mem::transmute(w.as_ptr()) };
        let mut want = vec![0u16; whole];
        unsafe { fw(g.as_ptr(), u.as_ptr(), want.as_mut_ptr()) };
        for dim in [4304usize, 4112] {
            let code = tailed(ElemSpec {
                op: ElemOp::GeluTanh,
                dim,
            })
            .unwrap();
            let f: SwiGluFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let mut y = vec![0x7fc1u16; dim + 40];
            unsafe { f(g.as_ptr(), u.as_ptr(), y.as_mut_ptr()) };
            assert_eq!(&y[..dim], &want[..dim], "dim {dim}");
            assert!(
                y[dim..].iter().all(|&v| v == 0x7fc1),
                "dim {dim}: written past the row"
            );
        }
    }

    /// `Standardize` bit for bit against `bf16((x · r − b) · c)` in f32, at 1 152 (whole blocks)
    /// and 72 (a partial).
    #[test]
    fn standardize_is_three_f32_steps() {
        for dim in [1152usize, 72] {
            let Some(code) = tailed(ElemSpec {
                op: ElemOp::Standardize,
                dim,
            }) else {
                return;
            };
            let f: RmsNormFn = unsafe { std::mem::transmute(code.as_ptr()) };
            let x = values(dim, 51);
            let mut w = values(dim, 53);
            w.extend(values(dim, 55));
            let r = 1152f32.sqrt();
            let mut y = vec![0x7fc1u16; dim + 40];
            unsafe { f(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), r) };
            for i in 0..dim {
                let want = bf16((f32_of(x[i]) * r - f32_of(w[i])) * f32_of(w[dim + i]));
                assert_eq!(y[i], want, "dim {dim} lane {i}");
            }
            assert!(
                y[dim..].iter().all(|&v| v == 0x7fc1),
                "dim {dim}: written past the row"
            );
        }
    }
}
