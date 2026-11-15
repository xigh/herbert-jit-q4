//! The AVX-512 attention kernels against a scalar reference, at `head_dim`
//! multiples of 32 and not (72, the vision tower of Gemma 4 26B-A4B): the row
//! kernel (chained and split, QK row by row and by groups of 16) and the block
//! kernel without mask (`full`), with a key count that is neither a multiple of
//! 16 nor of the block, every buffer followed by NaN and every output row by a
//! sentinel. The block must equal the row kernel bit for bit.
use crate::jit::emit_sdpa::{
    emit_sdpa, emit_sdpa_prefill, Pass, SdpaHeadFn, SdpaKernelSpec, SdpaPrefillFn, SdpaPrefillSpec,
    SdpaQkFn, SdpaSoftmaxFn, SdpaSvFn, SDPA_PF_DIST_PREFILL,
};
use crate::jit::ir::{ExpStrategy, SdpaForm, SdpaShape};

const NAN: u16 = 0x7FC0;
const SENTINEL: u16 = 0x1234;
/// NaN elements after each input buffer: more than the widest masked read
/// (a block of 32 bf16) can reach beyond a row.
const GUARD: usize = 64;

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}

/// `len` random bf16 in `[-amp, amp]`, then [`GUARD`] NaN.
fn guarded(len: usize, amp: f32, seed: &mut u64) -> Vec<u16> {
    let mut v: Vec<u16> = (0..len).map(|_| bf16(rng(seed) * amp)).collect();
    v.extend(std::iter::repeat_n(NAN, GUARD));
    v
}

/// Softmax attention of one query over `n` keys, scores and sums in f64.
fn reference(q: &[u16], k: &[u16], v: &[u16], n: usize, d: usize, scale: f32) -> Vec<f32> {
    let s: Vec<f64> = (0..n)
        .map(|j| {
            (0..d)
                .map(|i| f64::from(f(q[i])) * f64::from(f(k[j * d + i])))
                .sum::<f64>()
                * f64::from(scale)
        })
        .collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    (0..d)
        .map(|i| {
            ((0..n)
                .map(|j| e[j] * f64::from(f(v[j * d + i])))
                .sum::<f64>()
                / z) as f32
        })
        .collect()
}

/// Within four bf16 steps of the reference (the exp polynomial, the f32 sums
/// and the final rounding), with a floor for values near zero.
fn assert_close(got: &[u16], want: &[f32], what: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let step = w.abs().max(1.0 / 64.0) / 128.0;
        assert!(
            (f(g) - w).abs() <= 4.0 * step,
            "{what}: dimension {i}: {} against {w}",
            f(g)
        );
    }
}

fn shape(d: usize) -> SdpaShape {
    SdpaShape {
        head_dim: d,
        sink: false,
    }
}

/// The row kernel, `form` and `qk_group` given, on one query: its output (`d`
/// bf16), its sentinels checked.
fn run_row(
    spec: SdpaKernelSpec,
    q: &[u16],
    k: &[u16],
    v: &[u16],
    n: usize,
    scale: f32,
) -> Vec<u16> {
    let d = spec.shape.head_dim;
    let mut scores = vec![f32::NAN; n.div_ceil(16) * 16 + 16];
    let mut out = vec![SENTINEL; d + 32];
    match spec.form {
        SdpaForm::Split => {
            let blobs: Vec<_> = [Pass::Qk, Pass::Softmax, Pass::Sv]
                .iter()
                .map(|&p| exec(&emit_sdpa(&spec, Some(p)).unwrap()))
                .collect();
            unsafe {
                let qk: SdpaQkFn = std::mem::transmute(blobs[0].as_ptr());
                let sm: SdpaSoftmaxFn = std::mem::transmute(blobs[1].as_ptr());
                let sv: SdpaSvFn = std::mem::transmute(blobs[2].as_ptr());
                qk(q.as_ptr(), k.as_ptr(), scores.as_mut_ptr(), n as u64, scale);
                sm(scores.as_mut_ptr(), n as u64, 0.0);
                sv(scores.as_ptr(), v.as_ptr(), out.as_mut_ptr(), n as u64);
            }
        }
        _ => {
            let page = exec(&emit_sdpa(&spec, None).unwrap());
            unsafe {
                let run: SdpaHeadFn = std::mem::transmute(page.as_ptr());
                run(
                    q.as_ptr(),
                    k.as_ptr(),
                    v.as_ptr(),
                    scores.as_mut_ptr(),
                    out.as_mut_ptr(),
                    n as u64,
                    scale,
                    0.0,
                );
            }
        }
    }
    assert!(
        out[d..].iter().all(|&x| x == SENTINEL),
        "row kernel wrote beyond head_dim {d}"
    );
    out.truncate(d);
    out
}

fn row_spec(d: usize, form: SdpaForm, qk_group: u8) -> SdpaKernelSpec {
    SdpaKernelSpec {
        shape: shape(d),
        form,
        qk_group,
        ..SdpaKernelSpec::default()
    }
}

#[test]
fn row_kernel_matches_reference() {
    if !crate::jit::gemm::is_supported() {
        eprintln!("no AVX-512 BF16: skipped");
        return;
    }
    let mut seed = 7u64;
    for d in [64usize, 72, 80, 88, 128] {
        for n in [1usize, 15, 37, 53] {
            let q = guarded(d, 1.0, &mut seed);
            let k = guarded(n * d, 1.0, &mut seed);
            let v = guarded(n * d, 1.0, &mut seed);
            let scale = 1.0 / (d as f32).sqrt();
            let want = reference(&q, &k, &v, n, d, scale);
            for form in [SdpaForm::Chained, SdpaForm::Split] {
                for g in [0u8, 8, 16] {
                    let got = run_row(row_spec(d, form, g), &q, &k, &v, n, scale);
                    assert_close(&got, &want, &format!("row d={d} n={n} {form:?} qk={g}"));
                }
            }
        }
    }
}

/// The SV accumulators on the stack scratch, the partial vector among them: at
/// 72 all of them (`sv_spill_all`), at 440 and 504 the ones beyond the 27
/// registers. QK by 16 does not fit at 504 (16 + 16 + 3 > 32) and declines.
#[test]
fn spilled_sv_matches_reference() {
    if !crate::jit::gemm::is_supported() {
        eprintln!("no AVX-512 BF16: skipped");
        return;
    }
    assert!(row_spec(504, SdpaForm::Chained, 16).check().is_err());
    let mut seed = 5u64;
    for d in [72usize, 440, 504] {
        let n = 37;
        let q = guarded(d, 1.0, &mut seed);
        let k = guarded(n * d, 1.0, &mut seed);
        let v = guarded(n * d, 1.0, &mut seed);
        let scale = 1.0 / (d as f32).sqrt();
        let want = reference(&q, &k, &v, n, d, scale);
        for form in [SdpaForm::Chained, SdpaForm::Split] {
            for g in [0u8, 8] {
                for spill in [false, true] {
                    let spec = SdpaKernelSpec {
                        sv_spill_all: spill,
                        ..row_spec(d, form, g)
                    };
                    assert!(
                        spec.check().is_ok(),
                        "d={d} {form:?} qk={g} spill={spill} declined"
                    );
                    let got = run_row(spec, &q, &k, &v, n, scale);
                    assert_close(
                        &got,
                        &want,
                        &format!("row d={d} n={n} {form:?} qk={g} spill={spill}"),
                    );
                }
            }
        }
    }
}

#[test]
fn fused_and_other_widths_decline() {
    // The fused form keeps whole blocks of 32; a multiple of 8 only.
    assert!(row_spec(72, SdpaForm::Fused, 0).check().is_err());
    assert!(row_spec(68, SdpaForm::Chained, 0).check().is_err());
    assert!(row_spec(72, SdpaForm::Chained, 16).check().is_ok());
}

fn block_spec(
    d: usize,
    rows: usize,
    qk_group: u8,
    qk_pair: u8,
    n: usize,
    full: bool,
) -> SdpaPrefillSpec {
    SdpaPrefillSpec {
        shape: shape(d),
        rows,
        row_stride: d,
        out_row_stride: d + 8,
        exp: ExpStrategy::Inline,
        pf_dist: SDPA_PF_DIST_PREFILL,
        dim_blocks: 0,
        qk_group,
        qk_pair,
        kv_capacity: n,
        full,
        window: 0,
        sv_tile: 0,
    }
}

/// The block kernel: `rows` queries over `n` keys, all of them without mask
/// (`full`), the first `n - rows + i + 1` for row `i` when causal. Equal bit
/// for bit to the row kernel (chained, the same QK grouping) on each row, and
/// close to the reference. Queries paired by 2 cover a whole last packet
/// (8 rows) and a lone one (3 and 5).
#[test]
fn block_matches_row_kernel() {
    if !crate::jit::gemm::is_supported() {
        eprintln!("no AVX-512 BF16: skipped");
        return;
    }
    let mut seed = 11u64;
    for d in [64usize, 72, 80, 88, 128] {
        for (rows, n) in [(8usize, 37usize), (3, 53), (8, 8), (5, 21)] {
            for (g, pair) in [(16u8, 1u8), (8, 1), (8, 2), (0, 1)] {
                for full in [true, false] {
                    let spec = block_spec(d, rows, g, pair, n, full);
                    assert!(
                        spec.check().is_ok(),
                        "d={d} rows={rows} qk={g} pair={pair} declined"
                    );
                    let q = guarded(rows * d, 1.0, &mut seed);
                    let k = guarded(n * d, 1.0, &mut seed);
                    let v = guarded(n * d, 1.0, &mut seed);
                    let scale = 1.0 / (d as f32).sqrt();
                    let ors = spec.out_row_stride;
                    let mut scores = vec![f32::NAN; spec.scores_len(n) + 16];
                    let mut out = vec![SENTINEL; rows * ors + 32];
                    let page = exec(&emit_sdpa_prefill(&spec).unwrap());
                    // Without mask the kernel takes `N`, causal it takes `past`.
                    let arg = if full { n } else { n - rows };
                    unsafe {
                        let run: SdpaPrefillFn = std::mem::transmute(page.as_ptr());
                        run(
                            q.as_ptr(),
                            k.as_ptr(),
                            v.as_ptr(),
                            scores.as_mut_ptr(),
                            out.as_mut_ptr(),
                            arg as u64,
                            scale,
                            0.0,
                        );
                    }
                    let what = format!(
                        "block d={d} rows={rows} n={n} qk={g} pair={pair} full={full} nb={}",
                        spec.sv_dim_blocks()
                    );
                    for i in 0..rows {
                        let seen = if full { n } else { n - rows + i + 1 };
                        let row = &out[i * ors..i * ors + d];
                        assert!(
                            out[i * ors + d..(i + 1) * ors]
                                .iter()
                                .all(|&x| x == SENTINEL),
                            "{what}: row {i} wrote beyond head_dim"
                        );
                        let qi = &q[i * d..];
                        assert_close(
                            row,
                            &reference(qi, &k, &v, seen, d, scale),
                            &format!("{what} row {i}"),
                        );
                        let line =
                            run_row(row_spec(d, SdpaForm::Chained, g), qi, &k, &v, seen, scale);
                        assert_eq!(
                            row,
                            &line[..],
                            "{what}: row {i} differs from the row kernel"
                        );
                    }
                    assert!(
                        out[rows * ors..].iter().all(|&x| x == SENTINEL),
                        "{what}: wrote beyond the block"
                    );
                }
            }
        }
    }
}

/// The SV pass's groups and tiles (`dim_blocks`, `sv_tile`) at `head_dim` 72 and around it: every
/// row of a block, rows 1 to 8, equal bit for bit to the row kernel. Groups that do not divide
/// `head_dim/16` (72: `3 + 2`, `2 + 2 + 1`, `4 + 1`), tiles of 16 and 128 positions over 37 keys (a
/// partial tile) and 300 (two whole tiles of 128 and a third of 44; eighteen of 16 and one of 12),
/// without tile (`1`) and the shape's default (`0`).
#[test]
fn block_sv_groups_and_tiles_match_row_kernel() {
    if !crate::jit::gemm::is_supported() {
        eprintln!("no AVX-512 BF16: skipped");
        return;
    }
    let mut seed = 13u64;
    let mut cases = 0usize;
    for d in [72usize, 64, 88, 104, 128] {
        let nv = d.div_ceil(16);
        for n in [37usize, 300] {
            let q = guarded(8 * d, 1.0, &mut seed);
            let k = guarded(n * d, 1.0, &mut seed);
            let v = guarded(n * d, 1.0, &mut seed);
            let scale = 1.0 / (d as f32).sqrt();
            for g in [16u8, 8] {
                // The row kernel's output per (row, keys seen), computed once.
                let mut lines = std::collections::HashMap::new();
                for rows in 1..=8usize {
                    for nb in 0..=nv {
                        for tile in [0usize, 1, 16, 128] {
                            for full in [true, false] {
                                let spec = SdpaPrefillSpec {
                                    dim_blocks: nb,
                                    sv_tile: tile,
                                    ..block_spec(d, rows, g, 1, n, full)
                                };
                                if spec.check().is_err() {
                                    assert!(nb > 0, "d={d} rows={rows} qk={g} default declined");
                                    continue;
                                }
                                cases += 1;
                                let ors = spec.out_row_stride;
                                let mut scores = vec![f32::NAN; spec.scores_len(n) + 16];
                                let mut out = vec![SENTINEL; rows * ors + 32];
                                let page = exec(&emit_sdpa_prefill(&spec).unwrap());
                                let arg = if full { n } else { n - rows };
                                unsafe {
                                    let run: SdpaPrefillFn = std::mem::transmute(page.as_ptr());
                                    run(
                                        q.as_ptr(),
                                        k.as_ptr(),
                                        v.as_ptr(),
                                        scores.as_mut_ptr(),
                                        out.as_mut_ptr(),
                                        arg as u64,
                                        scale,
                                        0.0,
                                    );
                                }
                                let what = format!(
                                    "block d={d} rows={rows} n={n} qk={g} full={full} groups={:?} tile={:?}",
                                    spec.sv_groups(),
                                    spec.sv_tile_len()
                                );
                                for i in 0..rows {
                                    let seen = if full { n } else { n - rows + i + 1 };
                                    let row = &out[i * ors..i * ors + d];
                                    assert!(
                                        out[i * ors + d..(i + 1) * ors]
                                            .iter()
                                            .all(|&x| x == SENTINEL),
                                        "{what}: row {i} wrote beyond head_dim"
                                    );
                                    let line = lines.entry((i, seen)).or_insert_with(|| {
                                        run_row(
                                            row_spec(d, SdpaForm::Chained, g),
                                            &q[i * d..],
                                            &k,
                                            &v,
                                            seen,
                                            scale,
                                        )
                                    });
                                    assert_eq!(
                                        row,
                                        &line[..],
                                        "{what}: row {i} differs from the row kernel"
                                    );
                                }
                                assert!(
                                    out[rows * ors..].iter().all(|&x| x == SENTINEL),
                                    "{what}: wrote beyond the block"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("{cases} blocks equal to the row kernel");
}
