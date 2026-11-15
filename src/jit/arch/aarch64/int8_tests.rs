//! The int8 quantizer and GEMM kernel against a scalar reference (`jit/int8.rs`).
use crate::jit::emit::{AFormat, KernelOptions, MicroKernelSpec, NeonOptions};
use crate::jit::execution::abi::PackCall;
use crate::jit::int8::{self, a_row_bytes, GROUP};

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

/// The reference quantization of one group: scale and codes.
fn quant_ref(x: &[f32]) -> (f32, Vec<i8>) {
    let max = x.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0e-20);
    let inv = 127.0 / max;
    (
        max * (1.0 / 127.0),
        x.iter()
            .map(|v| (v * inv).round_ties_even() as i8)
            .collect(),
    )
}

fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}

#[test]
fn quantizer_matches_reference() {
    let (rows, k) = (3usize, 6144usize);
    let mut seed = 7u64;
    let mut src: Vec<u16> = (0..rows * k)
        .map(|i| bf16(rng(&mut seed) * if i / 32 % 3 == 0 { 100.0 } else { 0.01 }))
        .collect();
    for v in &mut src[32..64] {
        *v = 0;
    }
    let mut dest = vec![0u8; rows * a_row_bytes(k)];
    let kernel = super::execution::int8::quantize(k).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "C" fn(*const PackCall, usize) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let n = rows;
    let call = PackCall {
        source: src.as_ptr(),
        dest: dest.as_mut_ptr().cast(),
        rows: &n,
    };
    for r in 0..rows {
        unsafe { run(&call, r) };
    }
    for r in 0..rows {
        let row = &dest[r * a_row_bytes(k)..(r + 1) * a_row_bytes(k)];
        for g in 0..k / GROUP {
            let x: Vec<f32> = src[r * k + g * 32..r * k + g * 32 + 32]
                .iter()
                .map(|&b| f(b))
                .collect();
            let (scale, codes) = quant_ref(&x);
            let got = f32::from_le_bytes(row[k + 4 * g..k + 4 * g + 4].try_into().unwrap());
            assert!(
                (got - scale).abs() <= scale * 1e-6,
                "row {r} group {g}: scale {got} vs {scale}"
            );
            for i in 0..32 {
                assert_eq!(
                    row[g * 32 + i] as i8,
                    codes[i],
                    "row {r} group {g} value {i}"
                );
            }
        }
    }
}

#[test]
fn gemm_matches_reference() {
    let (n, k) = (40usize, 96usize);
    let mut seed = 11u64;
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = int8::pack(&w, n, k);
    for mr in 1..=2usize {
        let a_src: Vec<u16> = (0..mr * k).map(|_| bf16(rng(&mut seed) * 3.0)).collect();
        let mut a_q = vec![0u8; mr * a_row_bytes(k)];
        let kernel = super::execution::int8::quantize(k).unwrap();
        let page = exec(&kernel.bytes);
        let quant: unsafe extern "C" fn(*const PackCall, usize) =
            unsafe { std::mem::transmute(page.as_ptr()) };
        let rows = mr;
        let call = PackCall {
            source: a_src.as_ptr(),
            dest: a_q.as_mut_ptr().cast(),
            rows: &rows,
        };
        for r in 0..mr {
            unsafe { quant(&call, r) };
        }
        for out_bf16 in [false] {
            let mut c = vec![0f32; mr * n.div_ceil(16) * 16];
            for block in 0..n.div_ceil(16) {
                let n_tail = if (block + 1) * 16 > n {
                    n - block * 16
                } else {
                    16
                };
                let spec = MicroKernelSpec {
                    mr,
                    nr: 16,
                    k_pairs: k / 2,
                    k_odd: false,
                    lda_bytes: a_row_bytes(k),
                    a_fmt: AFormat::Bf16,
                    ldc_bytes: n.div_ceil(16) * 16 * 4,
                    n_tail,
                    out_bf16,
                    opts: KernelOptions {
                        neon: NeonOptions {
                            w_int8: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    k_dynamic: false,
                };
                let bytes = super::gemm_int8::emit(&spec).unwrap();
                let page = exec(&bytes);
                let run: unsafe extern "C" fn(*const u8, *const u8, *mut f32) =
                    unsafe { std::mem::transmute(page.as_ptr()) };
                unsafe {
                    run(
                        a_q.as_ptr(),
                        packed.as_ptr().add(block * int8::block_bytes(k, false)),
                        c.as_mut_ptr().add(block * 16),
                    )
                };
            }
            for i in 0..mr {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for g in 0..k / GROUP {
                        let arow = &a_q[i * a_row_bytes(k)..];
                        let sa =
                            f32::from_le_bytes(arow[k + 4 * g..k + 4 * g + 4].try_into().unwrap());
                        let (block, lane) = (j / 16, j % 16);
                        let base = block * int8::block_bytes(k, false) + g * int8::GROUP_BYTES;
                        let sw = f32::from_le_bytes(
                            packed[base + 512 + 4 * lane..base + 512 + 4 * lane + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let mut dot = 0i32;
                        for kk in 0..32 {
                            let qw = packed
                                [base + (kk / 4) * 64 + (lane / 4) * 16 + (lane % 4) * 4 + kk % 4]
                                as i8;
                            dot += i32::from(qw) * i32::from(arow[g * 32 + kk] as i8);
                        }
                        acc += f64::from(sa) * f64::from(sw) * f64::from(dot);
                    }
                    let got = c[i * n.div_ceil(16) * 16 + j];
                    assert!(
                        (f64::from(got) - acc).abs() <= 1e-4 * acc.abs().max(1.0),
                        "mr {mr} C[{i}][{j}] = {got}, reference {acc}"
                    );
                }
            }
        }
    }
}

#[test]
fn q4_gemm_matches_reference() {
    let (n, k) = (40usize, 96usize);
    let mut seed = 5u64;
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = int8::pack_q4(&w, n, k);
    let block = k / GROUP * int8::Q4_GROUP_BYTES;
    for mr in 1..=2usize {
        let a_src: Vec<u16> = (0..mr * k).map(|_| bf16(rng(&mut seed) * 3.0)).collect();
        let mut a_q = vec![0u8; mr * a_row_bytes(k)];
        let kernel = super::execution::int8::quantize(k).unwrap();
        let page = exec(&kernel.bytes);
        let quant: unsafe extern "C" fn(*const PackCall, usize) =
            unsafe { std::mem::transmute(page.as_ptr()) };
        let rows = mr;
        let call = PackCall {
            source: a_src.as_ptr(),
            dest: a_q.as_mut_ptr().cast(),
            rows: &rows,
        };
        for r in 0..mr {
            unsafe { quant(&call, r) };
        }
        let mut c = vec![0f32; mr * n.div_ceil(16) * 16];
        for b in 0..n.div_ceil(16) {
            let n_tail = if (b + 1) * 16 > n { n - b * 16 } else { 16 };
            let spec = MicroKernelSpec {
                mr,
                nr: 16,
                k_pairs: k / 2,
                k_odd: false,
                lda_bytes: a_row_bytes(k),
                a_fmt: AFormat::Bf16,
                ldc_bytes: n.div_ceil(16) * 16 * 4,
                n_tail,
                out_bf16: false,
                opts: KernelOptions {
                    neon: NeonOptions {
                        w_int8: true,
                        w_q4: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                k_dynamic: false,
            };
            let bytes = super::gemm_q4::emit(&spec).unwrap();
            let page = exec(&bytes);
            let run: unsafe extern "C" fn(*const u8, *const u8, *mut f32) =
                unsafe { std::mem::transmute(page.as_ptr()) };
            unsafe {
                run(
                    a_q.as_ptr(),
                    packed.as_ptr().add(b * block),
                    c.as_mut_ptr().add(b * 16),
                )
            };
        }
        for i in 0..mr {
            for j in 0..n {
                let mut acc = 0.0f64;
                for g in 0..k / GROUP {
                    let arow = &a_q[i * a_row_bytes(k)..];
                    let sa = f32::from_le_bytes(arow[k + 4 * g..k + 4 * g + 4].try_into().unwrap());
                    let (b, lane) = (j / 16, j % 16);
                    let base = b * block + g * int8::Q4_GROUP_BYTES;
                    let sw = f32::from_le_bytes(
                        packed[base + 256 + 4 * lane..base + 256 + 4 * lane + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let mut dot = 0i32;
                    for kk in 0..32 {
                        let byte = packed
                            [base + (kk / 8) * 64 + (lane / 4) * 16 + (lane % 4) * 4 + kk % 4];
                        let nib = if kk % 8 >= 4 { byte >> 4 } else { byte & 0xf };
                        let code = ((nib << 4) as i8) >> 4;
                        dot += i32::from(code) * 16 * i32::from(arow[g * 32 + kk] as i8);
                    }
                    acc += f64::from(sa) * f64::from(sw) * f64::from(dot);
                }
                let got = c[i * n.div_ceil(16) * 16 + j];
                assert!(
                    (f64::from(got) - acc).abs() <= 1e-4 * acc.abs().max(1.0),
                    "mr {mr} C[{i}][{j}] = {got}, reference {acc}"
                );
            }
        }
    }
}

/// The Q4 kernel of four rows against the kernel of two, bit for bit: every row of a tile of 1 to 4
/// rows against the kernel of two run on that row alone, f32 and bf16 out, full and partial blocks
/// of columns.
#[test]
fn q4_four_rows_match_two() {
    let (n, k) = (40usize, 1024usize);
    let mut seed = 31u64;
    let w: Vec<u16> = (0..n * k)
        .map(|i| bf16(rng(&mut seed) * if i % 97 == 0 { 40.0 } else { 1.0 }))
        .collect();
    let packed = int8::pack_q4(&w, n, k);
    let block = k / GROUP * int8::Q4_GROUP_BYTES;
    let kernel = super::execution::int8::quantize(k).unwrap();
    let qpage = exec(&kernel.bytes);
    let quant: unsafe extern "C" fn(*const PackCall, usize) =
        unsafe { std::mem::transmute(qpage.as_ptr()) };
    let width = n.div_ceil(16) * 16;
    for mr in 1..=super::gemm_q4::ROWS4 {
        for out_bf16 in [false, true] {
            let a_src: Vec<u16> = (0..mr * k)
                .map(|i| bf16(rng(&mut seed) * if i % 53 == 0 { 30.0 } else { 2.0 }))
                .collect();
            let mut a_q = vec![0u8; mr * a_row_bytes(k)];
            let call = PackCall {
                source: a_src.as_ptr(),
                dest: a_q.as_mut_ptr().cast(),
                rows: &mr,
            };
            for r in 0..mr {
                unsafe { quant(&call, r) };
            }
            let es = if out_bf16 { 2 } else { 4 };
            let spec = |rows: usize, n_tail: usize| MicroKernelSpec {
                mr: rows,
                nr: 16,
                k_pairs: k / 2,
                k_odd: false,
                lda_bytes: a_row_bytes(k),
                a_fmt: AFormat::Bf16,
                ldc_bytes: width * es,
                n_tail,
                out_bf16,
                opts: KernelOptions {
                    neon: NeonOptions {
                        w_int8: true,
                        w_q4: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                k_dynamic: false,
            };
            let mut four = vec![0u8; mr * width * es];
            let mut two = vec![0u8; mr * width * es];
            for b in 0..n.div_ceil(16) {
                let n_tail = if (b + 1) * 16 > n { n - b * 16 } else { 16 };
                let k4 = super::gemm_q4::emit4(&spec(mr, n_tail)).unwrap();
                let p4 = exec(&k4);
                let run4: unsafe extern "C" fn(*const u8, *const u8, *mut u8) =
                    unsafe { std::mem::transmute(p4.as_ptr()) };
                unsafe {
                    run4(
                        a_q.as_ptr(),
                        packed.as_ptr().add(b * block),
                        four.as_mut_ptr().add(b * 16 * es),
                    )
                };
                let k2 = super::gemm_q4::emit2(&spec(1, n_tail)).unwrap();
                let p2 = exec(&k2);
                let run2: unsafe extern "C" fn(*const u8, *const u8, *mut u8) =
                    unsafe { std::mem::transmute(p2.as_ptr()) };
                for i in 0..mr {
                    unsafe {
                        run2(
                            a_q.as_ptr().add(i * a_row_bytes(k)),
                            packed.as_ptr().add(b * block),
                            two.as_mut_ptr().add((i * width + b * 16) * es),
                        )
                    };
                }
            }
            for i in 0..mr {
                let row = |c: &[u8]| c[i * width * es..(i * width + n) * es].to_vec();
                assert_eq!(row(&four), row(&two), "mr {mr} bf16 {out_bf16} row {i}");
            }
        }
    }
}

/// `cargo test --release --lib q4_rows_speed -- --ignored --nocapture`: the Q4 kernel of four rows
/// against the one of two, 120 rows of `A` against sixteen columns (weights in L1).
#[test]
#[ignore = "a speed measurement, run by hand"]
fn q4_rows_speed() {
    let (n, k, reps) = (16usize, 1024usize, 4_000usize);
    let mut seed = 3u64;
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = int8::pack_q4(&w, n, k);
    let rows = 120usize;
    let a_src: Vec<u16> = (0..rows * k).map(|_| bf16(rng(&mut seed))).collect();
    let mut a_q = vec![0u8; rows * a_row_bytes(k)];
    let kernel = super::execution::int8::quantize(k).unwrap();
    let qpage = exec(&kernel.bytes);
    let quant: unsafe extern "C" fn(*const PackCall, usize) =
        unsafe { std::mem::transmute(qpage.as_ptr()) };
    let call = PackCall {
        source: a_src.as_ptr(),
        dest: a_q.as_mut_ptr().cast(),
        rows: &rows,
    };
    for r in 0..rows {
        unsafe { quant(&call, r) };
    }
    let mut c = vec![0u16; rows * n];
    for (label, mr, four) in [("two rows", 2usize, false), ("four rows", 4, true)] {
        let spec = MicroKernelSpec {
            mr,
            nr: 16,
            k_pairs: k / 2,
            k_odd: false,
            lda_bytes: a_row_bytes(k),
            a_fmt: AFormat::Bf16,
            ldc_bytes: n * 2,
            n_tail: 16,
            out_bf16: true,
            opts: KernelOptions {
                neon: NeonOptions {
                    w_int8: true,
                    w_q4: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            k_dynamic: false,
        };
        let bytes = if four {
            super::gemm_q4::emit4(&spec).unwrap()
        } else {
            super::gemm_q4::emit2(&spec).unwrap()
        };
        let page = exec(&bytes);
        let run: unsafe extern "C" fn(*const u8, *const u8, *mut u8) =
            unsafe { std::mem::transmute(page.as_ptr()) };
        let mut best = std::time::Duration::from_secs(3600);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for r in (0..rows).step_by(mr) {
                    unsafe {
                        run(
                            a_q.as_ptr().add(r * a_row_bytes(k)),
                            packed.as_ptr(),
                            c.as_mut_ptr().add(r * n).cast(),
                        )
                    };
                }
            }
            best = best.min(t.elapsed());
        }
        println!(
            "{label}: {:.1} GMAC/s",
            (rows * n * k * reps) as f64 / best.as_secs_f64() / 1e9
        );
    }
}
