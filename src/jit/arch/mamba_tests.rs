//! Mamba2's emitted kernels (Granite 4) against a scalar reference, on the target of the build
//! (x86-64, AArch64, RISC-V): the gates, the conv with its biases, the recurrence (on x86 the
//! emitted scalar and the AVX-512 one), the gated norm over the row.
use crate::jit::arch::{KernelSet, Native};
use crate::jit::execution::abi::{ConvCall, GateCall, NormCall, RecurrenceCall, Shape};
use crate::jit::execution::math_data::TABLES;

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}
fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}
fn f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}
fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}
type Task<C> = crate::jit::arch::generated_fn!(*const C, usize);
fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[test]
fn gates_match_reference() {
    let (rows, nv) = (3usize, 48usize);
    let mut seed = 1u64;
    let ab: Vec<f32> = (0..rows * nv).map(|_| rng(&mut seed) * 4.0).collect();
    let a_log: Vec<f32> = (0..nv).map(|_| rng(&mut seed)).collect();
    let bias: Vec<u16> = (0..nv).map(|_| bf16(rng(&mut seed))).collect();
    let (mut beta, mut decay) = (vec![0f32; rows * nv], vec![0f32; rows * nv]);
    let call = GateCall {
        ab: ab.as_ptr(),
        a_log: a_log.as_ptr(),
        bias: bias.as_ptr(),
        beta: beta.as_mut_ptr(),
        decay: decay.as_mut_ptr(),
        tables: &TABLES,
    };
    let k = Native::mamba_gates(nv, true).unwrap();
    let page = exec(&k.bytes);
    let run: Task<GateCall> = unsafe { std::mem::transmute(page.as_ptr()) };
    for t in 0..rows * nv {
        unsafe { run(&call, t) };
    }
    for t in 0..rows * nv {
        let h = t % nv;
        let dt = softplus(ab[t] + f(bias[h]));
        assert!(close(beta[t], dt, 1e-5), "beta[{t}] {} vs {dt}", beta[t]);
        let d = (-(a_log[h].exp()) * dt).exp();
        assert!(close(decay[t], d, 1e-5), "decay[{t}] {} vs {d}", decay[t]);
    }
}

#[test]
fn conv_matches_reference() {
    let shape = Shape {
        nk: 1,
        nv: 48,
        hkd: 128,
        hvd: 64,
        ck: 4,
    };
    let (conv, ck, rows) = (shape.conv(), shape.ck, 5usize);
    let mut seed = 2u64;
    let input: Vec<f32> = (0..rows * conv).map(|_| rng(&mut seed)).collect();
    let mut data = input.clone();
    let w: Vec<u16> = (0..conv * ck).map(|_| bf16(rng(&mut seed))).collect();
    let bias: Vec<f32> = (0..conv).map(|_| rng(&mut seed)).collect();
    let mut state = vec![0f32; conv * ck];
    let rows_n = rows;
    let call = ConvCall {
        data: data.as_mut_ptr(),
        weights: w.as_ptr(),
        state: state.as_mut_ptr(),
        rows: &rows_n,
        tables: &TABLES,
        bias: bias.as_ptr(),
    };
    let k = Native::mamba_convolution(shape, true).unwrap();
    let page = exec(&k.bytes);
    let run: Task<ConvCall> = unsafe { std::mem::transmute(page.as_ptr()) };
    for t in 0..conv.div_ceil(16) {
        unsafe { run(&call, t) };
    }
    for r in 0..rows {
        for c in 0..conv {
            let mut s = 0f32;
            for j in 0..ck {
                // tap j reads row r − (ck − 1 − j)
                let src = r as isize - (ck as isize - 1 - j as isize);
                if src >= 0 {
                    s += f(w[c * ck + j]) * input[src as usize * conv + c];
                }
            }
            s += bias[c];
            let y = s / (1.0 + (-s).exp());
            assert!(
                close(data[r * conv + c], y, 1e-5),
                "row {r} channel {c}: {} vs {y}",
                data[r * conv + c]
            );
        }
    }
}

fn recurrence_case(vector: bool) {
    let shape = Shape {
        nk: 1,
        nv: 48,
        hkd: 128,
        hvd: 64,
        ck: 4,
    };
    let (key, value, conv, nv, hkd, hvd, rows) = (
        shape.key(),
        shape.value(),
        shape.conv(),
        shape.nv,
        shape.hkd,
        shape.hvd,
        4usize,
    );
    let mut seed = 3u64;
    let qkv: Vec<f32> = (0..rows * conv).map(|_| rng(&mut seed)).collect();
    let beta: Vec<f32> = (0..rows * nv)
        .map(|_| 0.05 + rng(&mut seed).abs() * 0.1)
        .collect();
    let decay: Vec<f32> = (0..rows * nv)
        .map(|_| 0.8 + rng(&mut seed).abs() * 0.2)
        .collect();
    let d: Vec<f32> = (0..nv).map(|_| rng(&mut seed)).collect();
    let elem16 = !crate::jit::knobs::knobs().deltanet_state_f32;
    let mut state = vec![0f32; nv * hkd * hvd];
    let mut out = vec![0f32; rows * value];
    let (rows_n, pos) = (rows, 0usize);
    let call = RecurrenceCall {
        state: state.as_mut_ptr(),
        output: out.as_mut_ptr(),
        qkv: qkv.as_ptr(),
        beta: beta.as_ptr(),
        decay: decay.as_ptr(),
        rows: &rows_n,
        position: &pos,
        prefill: 1,
        d: d.as_ptr(),
    };
    let compiled = if vector {
        let spec = crate::jit::emit_recurrence::RecurrenceSpec {
            mamba: true,
            ..crate::jit::recurrence::spec_of_knobs(hvd)
        };
        crate::jit::recurrence::get_or_compile(spec).unwrap()
    } else {
        None
    };
    if vector && compiled.is_none() && cfg!(target_arch = "x86_64") {
        return;
    }
    let k = Native::mamba_recurrence(shape, compiled.as_deref()).unwrap();
    let page = exec(&k.bytes);
    let run: Task<RecurrenceCall> = unsafe { std::mem::transmute(page.as_ptr()) };
    for h in 0..nv {
        unsafe { run(&call, h) };
    }
    let mut s = vec![0f32; nv * hkd * hvd];
    for r in 0..rows {
        let row = &qkv[r * conv..(r + 1) * conv];
        for h in 0..nv {
            let (c, b, x) = (
                &row[..key],
                &row[key..2 * key],
                &row[2 * key + h * hvd..2 * key + (h + 1) * hvd],
            );
            let (dt, dec) = (beta[r * nv + h], decay[r * nv + h]);
            for p in 0..hvd {
                let mut y = 0f32;
                for n in 0..hkd {
                    let cell = &mut s[h * hkd * hvd + n * hvd + p];
                    *cell = *cell * dec + (b[n] * dt) * x[p];
                    y += *cell * c[n];
                    if elem16 {
                        *cell = f(
                            ((cell.to_bits() + 0x7fff + ((cell.to_bits() >> 16) & 1)) >> 16) as u16,
                        );
                    }
                }
                y += x[p] * d[h];
                let got = out[r * value + h * hvd + p];
                assert!(
                    close(got, y, 2e-3),
                    "vector {vector} row {r} head {h} p {p}: {got} vs {y}"
                );
            }
        }
    }
}

#[test]
fn recurrence_scalar_matches_reference() {
    recurrence_case(false);
}

#[test]
fn recurrence_vector_matches_reference() {
    recurrence_case(true);
}

#[test]
fn gated_norm_matches_reference() {
    let (rows, dim, eps) = (2usize, 3072usize, 1e-5f32);
    let mut seed = 4u64;
    let y0: Vec<f32> = (0..rows * dim).map(|_| rng(&mut seed) * 3.0).collect();
    let mut y = y0.clone();
    let z: Vec<f32> = (0..rows * dim).map(|_| rng(&mut seed) * 3.0).collect();
    let w: Vec<u16> = (0..dim).map(|_| bf16(rng(&mut seed))).collect();
    let mut out = vec![0u16; rows * dim];
    let call = NormCall {
        output: y.as_mut_ptr(),
        gate: z.as_ptr(),
        weights: w.as_ptr(),
        output_bf16: out.as_mut_ptr(),
        tables: &TABLES,
        eps,
    };
    let k = Native::mamba_normed_gate(dim, true).unwrap();
    let page = exec(&k.bytes);
    let run: Task<NormCall> = unsafe { std::mem::transmute(page.as_ptr()) };
    for r in 0..rows {
        unsafe { run(&call, r) };
    }
    for r in 0..rows {
        let g: Vec<f32> = (0..dim)
            .map(|i| y0[r * dim + i] * (z[r * dim + i] / (1.0 + (-z[r * dim + i]).exp())))
            .collect();
        let ms = g.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let rr = 1.0 / (ms + eps).sqrt();
        for i in 0..dim {
            let want = f(w[i]) * (g[i] * rr);
            assert!(
                close(f(out[r * dim + i]), want, 1e-2),
                "row {r} i {i}: {} vs {want}",
                f(out[r * dim + i])
            );
        }
    }
}
