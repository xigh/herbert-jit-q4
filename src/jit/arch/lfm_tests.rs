//! LiquidAI's emitted kernels (LFM2, `lfm2_moe`, LFM3) against a scalar reference, on the target of
//! the build (x86-64, AArch64): the gated short convolution (every bit, the rounding of each step
//! to bf16 included), the sigmoid route with its bias per expert (the experts exactly, the weights
//! to one bf16 step: the emitted exponential is a polynomial), and LFM3's gate per head (every bit
//! against the same polynomial written out in scalars).
use crate::jit::arch::{KernelSet, Native};
use crate::jit::execution::abi::{RouteCall, RouteRule, ShortConvCall, MAX_TOP_K};

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}
/// bf16 rounding to nearest even, kept as f32.
fn rne(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}
fn bf16(x: f32) -> u16 {
    (rne(x).to_bits() >> 16) as u16
}
fn f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}
fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}
type Task<C> = crate::jit::arch::generated_fn!(*const C, usize);
type Once<C> = crate::jit::arch::generated_fn!(*const C);

/// The conv over two calls (a prompt of five rows, then one row: the state carried), against
/// `Lfm2ShortConv` written out in scalars, bit for bit: `bx = bf16(bf16(B)·bf16(x))`, `conv =
/// bf16(w_0·s_0 + …)` (one product then fused multiply-adds, oldest first), `y =
/// bf16(bf16(C)·conv)`.
#[test]
fn short_conv_matches_reference() {
    for (width, taps) in [(64usize, 3usize), (48, 7)] {
        let kernel = Native::short_conv(width, taps).unwrap();
        let page = exec(&kernel.bytes);
        let run: Task<ShortConvCall> = unsafe { std::mem::transmute(page.as_ptr()) };
        let mut seed = 7u64 + width as u64;
        let w: Vec<u16> = (0..width * taps).map(|_| bf16(rng(&mut seed))).collect();
        // Tap-major by blocks of sixteen, as the loader packs them.
        let tapped: Vec<u16> = (0..width / 16)
            .flat_map(|b| (0..taps).flat_map(move |j| (0..16).map(move |l| (b, j, l))))
            .map(|(b, j, l)| w[(b * 16 + l) * taps + j])
            .collect();
        let mut state = vec![0f32; width * taps];
        let mut ref_state = vec![vec![0f32; taps]; width];
        for rows in [5usize, 1] {
            let gen = |seed: &mut u64| -> Vec<f32> {
                (0..rows * width).map(|_| rng(seed) * 3.0).collect()
            };
            let (b, c, x) = (gen(&mut seed), gen(&mut seed), gen(&mut seed));
            let mut out = vec![0u16; rows * width];
            let call = ShortConvCall {
                b: b.as_ptr(),
                c: c.as_ptr(),
                x: x.as_ptr(),
                weights: tapped.as_ptr(),
                state: state.as_mut_ptr(),
                output: out.as_mut_ptr(),
                rows: &rows,
            };
            for t in 0..width / 16 {
                unsafe { run(&call, t) };
            }
            for r in 0..rows {
                for ch in 0..width {
                    let i = r * width + ch;
                    let bx = rne(rne(b[i]) * rne(x[i]));
                    let s = &mut ref_state[ch];
                    s.rotate_left(1);
                    s[taps - 1] = bx;
                    let mut acc = s[0] * f(w[ch * taps]);
                    for j in 1..taps {
                        acc = s[j].mul_add(f(w[ch * taps + j]), acc);
                    }
                    let y = bf16(rne(c[i]) * rne(acc));
                    assert_eq!(
                        out[i], y,
                        "width {width} taps {taps} rows {rows} row {r} channel {ch}"
                    );
                }
            }
        }
    }
}

/// `Lfm2MoeTopKRouter`: `s = bf16(σ(score))`, the `top_k` best of `s + bias` (the lowest expert on
/// a tie), their `s` divided by `bf16(bf16(Σs) + 1e-6)` (`norm`), times the scale, each step
/// rounded; the slots in increasing expert order.
#[test]
fn route_lfm_matches_reference() {
    let mut seed = 31u64;
    for (experts, top_k, norm, scale) in [
        (32usize, 4usize, true, 1.0f32),
        (64, 4, true, 1.0),
        (64, 8, false, 2.5),
    ] {
        let kernel = Native::moe_route_rule(
            experts,
            top_k,
            RouteRule::Lfm {
                norm,
                scale_bits: scale.to_bits(),
            },
        )
        .unwrap();
        let page = exec(&kernel.bytes);
        let run: Once<RouteCall> = unsafe { std::mem::transmute(page.as_ptr()) };
        let (mut exact, mut all) = (0usize, 0usize);
        for round in 0..200 {
            let scores: Vec<u16> = (0..experts).map(|_| bf16(rng(&mut seed) * 6.0)).collect();
            let mut bias: Vec<f32> = (0..experts).map(|_| rng(&mut seed) * 0.05).collect();
            if round % 7 == 0 {
                bias[5] = bias[9];
            }
            let gu_table: Vec<u64> = (0..experts as u64).map(|e| 1000 + e).collect();
            let dn_table: Vec<u64> = (0..experts as u64).map(|e| 5000 + e).collect();
            let mut gu_slots = [0u64; MAX_TOP_K];
            let mut dn_slots = [0u64; MAX_TOP_K];
            let mut weights = [0f32; MAX_TOP_K];
            let mut call = RouteCall {
                scores: scores.as_ptr(),
                per_expert_scale: bias.as_ptr(),
                weights_out: weights.as_mut_ptr(),
                gate_up_table: gu_table.as_ptr(),
                down_table: dn_table.as_ptr(),
                gate_up_slots: [std::ptr::null_mut(); MAX_TOP_K],
                down_slots: [std::ptr::null_mut(); MAX_TOP_K],
            };
            for k in 0..top_k {
                call.gate_up_slots[k] = &mut gu_slots[k];
                call.down_slots[k] = &mut dn_slots[k];
            }
            unsafe { run(&call) };
            // The reference, in f64 for σ.
            let sig: Vec<f32> = scores
                .iter()
                .map(|&v| rne((1.0 / (1.0 + (-(f(v) as f64)).exp())) as f32))
                .collect();
            let mut choice: Vec<(f32, usize)> =
                (0..experts).map(|e| (sig[e] + bias[e], e)).collect();
            choice.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let picked: Vec<usize> = choice[..top_k].iter().map(|c| c.1).collect();
            let den = rne(rne(picked.iter().map(|&e| sig[e]).sum::<f32>()) + 1e-6);
            let mut want: Vec<(usize, f32)> = picked
                .iter()
                .map(|&e| {
                    let w = if norm { rne(sig[e] / den) } else { sig[e] };
                    (e, if scale != 1.0 { rne(w * scale) } else { w })
                })
                .collect();
            want.sort_by_key(|p| p.0);
            // A sigmoid one bf16 step away can move a choice near a tie: compare the experts only
            // where the margin around the k-th choice is wider than that step.
            let margin = choice[top_k - 1].0 - choice[top_k].0;
            if margin > 1.0 / 128.0 {
                for k in 0..top_k {
                    assert_eq!(
                        gu_slots[k],
                        1000 + want[k].0 as u64,
                        "experts {experts} round {round} slot {k}"
                    );
                    assert_eq!(dn_slots[k], 5000 + want[k].0 as u64);
                    let (got, w) = (weights[k], want[k].1);
                    assert!(
                        (got - w).abs() <= w.abs() / 64.0,
                        "experts {experts} round {round} slot {k}: {got} vs {w}"
                    );
                    all += 1;
                    exact += usize::from(got.to_bits() == w.to_bits());
                }
            }
        }
        assert!(
            all > 0 && exact * 100 >= all * 95,
            "experts {experts}: {exact} of {all} weights exact"
        );
    }
}

/// The emitted exponential of the routers and of the gate per head, written out in scalars (`x ≤
/// 0`): `t = max(x, −87) · log2 e`, `n = ⌊t⌋`, `p` the polynomial in `f = t − n` by fused
/// multiply-adds, `2ⁿ · p` by adding `n` to the exponent bits.
fn emitted_exp(x: f32) -> f32 {
    let t = x.max(-87.0) * std::f32::consts::LOG2_E;
    let n = t.floor();
    let fr = t - n;
    let mut p = 0.001_333_355_8f32;
    for c in [
        0.009_618_129f32,
        0.055_504_11,
        0.240_226_5,
        std::f32::consts::LN_2,
        1.0,
    ] {
        p = p.mul_add(fr, c);
    }
    f32::from_bits(((n as i32) << 23).wrapping_add(p.to_bits() as i32) as u32)
}

/// LFM3's gate per head over two calls (a block of five rows, then one row), against
/// `Lfm3Attention` written out in scalars, bit for bit: `s = bf16(σ(g))` with the emitted
/// exponential (`1 / (1 + e)` for `g ≥ 0`, `e / (1 + e)` below, `e = exp(−|g|)`), `o = bf16(o ·
/// s)`. And `s` within one bf16 step of the exact sigmoid (f64).
#[test]
fn head_gate_matches_reference() {
    use crate::jit::execution::abi::HeadGateCall;
    for (heads, head_dim) in [(16usize, 128usize), (4, 64)] {
        let kernel = Native::head_gate(heads, head_dim).unwrap();
        let page = exec(&kernel.bytes);
        let run: Task<HeadGateCall> = unsafe { std::mem::transmute(page.as_ptr()) };
        let mut seed = 101u64 + heads as u64;
        for rows in [5usize, 1] {
            let width = heads * head_dim;
            let input: Vec<u16> = (0..rows * width)
                .map(|_| bf16(rng(&mut seed) * 4.0))
                .collect();
            // Logits over both signs and the far tails (a saturated sigmoid, an `exp` clamped at
            // −87).
            let gate: Vec<u16> = (0..rows * heads)
                .map(|i| {
                    bf16(match i % 5 {
                        0 => rng(&mut seed) * 120.0,
                        1 => 0.0,
                        _ => rng(&mut seed) * 8.0,
                    })
                })
                .collect();
            let mut out = input.clone();
            let call = HeadGateCall {
                output: out.as_mut_ptr(),
                gate: gate.as_ptr(),
                rows: &rows,
            };
            for h in 0..heads {
                unsafe { run(&call, h) };
            }
            for r in 0..rows {
                for h in 0..heads {
                    let g = f(gate[r * heads + h]);
                    let e = emitted_exp(-g.abs());
                    let s = rne(if g.to_bits() >> 31 == 0 {
                        1.0 / (e + 1.0)
                    } else {
                        e / (e + 1.0)
                    });
                    let exact = (1.0 / (1.0 + (-(g as f64)).exp())) as f32;
                    // Below −87 the exponential stops at e^−87: σ ≈ 1.6e-38 where the exact one is
                    // smaller still.
                    let floor = if g < -87.0 { 2e-38 } else { 0.0 };
                    assert!(
                        (s - rne(exact)).abs() <= (exact.abs() / 128.0).max(floor),
                        "heads {heads} row {r} head {h}: σ({g}) {s} vs {exact}"
                    );
                    for d in 0..head_dim {
                        let i = r * width + h * head_dim + d;
                        assert_eq!(out[i], bf16(f(input[i]) * s), "heads {heads} head_dim {head_dim} rows {rows} row {r} head {h} value {d}");
                    }
                }
            }
        }
    }
}
