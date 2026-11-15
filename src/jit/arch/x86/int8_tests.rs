//! The x86-64 int8 / Q4 quantizer and GEMM kernel against a scalar reference (`jit/int8.rs`).
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

fn quantize(src: &[u16], rows: usize, k: usize) -> Vec<u8> {
    let mut dest = vec![0u8; rows * a_row_bytes(k)];
    let kernel = super::execution::int8::quantize(k).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "sysv64" fn(*const PackCall, usize) =
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
    dest
}

#[test]
fn quantizer_matches_reference() {
    if !super::gemm_int8::available() {
        return;
    }
    let (rows, k) = (3usize, 6144usize);
    let mut seed = 7u64;
    let mut src: Vec<u16> = (0..rows * k)
        .map(|i| bf16(rng(&mut seed) * if i / 32 % 3 == 0 { 100.0 } else { 0.01 }))
        .collect();
    for v in &mut src[32..64] {
        *v = 0;
    }
    let dest = quantize(&src, rows, k);
    for r in 0..rows {
        let row = &dest[r * a_row_bytes(k)..(r + 1) * a_row_bytes(k)];
        for g in 0..k / GROUP {
            let x: Vec<f32> = src[r * k + g * 32..r * k + g * 32 + 32]
                .iter()
                .map(|&b| f(b))
                .collect();
            let (scale, codes) = quant_ref(&x);
            let got = f32::from_le_bytes(row[k + 8 * g..k + 8 * g + 4].try_into().unwrap());
            let sum = i32::from_le_bytes(row[k + 8 * g + 4..k + 8 * g + 8].try_into().unwrap());
            assert!(
                (got - scale).abs() <= scale * 1e-6,
                "row {r} group {g}: scale {got} vs {scale}"
            );
            assert_eq!(
                sum,
                codes.iter().map(|&c| i32::from(c)).sum::<i32>(),
                "row {r} group {g}: sum"
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

fn gemm_check(q4: bool) {
    // int8 without VNNI saturates `vpmaddubsw` (module documentation of `gemm_int8.rs`): Q4 only
    // there.
    if !super::gemm_int8::available()
        || !(q4 || crate::core::isa::feature("avxvnni") || super::gemm_int8::zmm())
    {
        return;
    }
    let (n, k) = (48usize, 96usize);
    let mut seed = if q4 { 5 } else { 11 };
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = if q4 {
        int8::pack_q4(&w, n, k)
    } else {
        int8::pack_int8(&w, n, k)
    };
    let block = k / GROUP * int8::group_bytes(q4);
    for mr in 1..=super::gemm_int8::max_rows() {
        for out_bf16 in [false, true] {
            let a_src: Vec<u16> = (0..mr * k).map(|_| bf16(rng(&mut seed) * 3.0)).collect();
            let a_q = quantize(&a_src, mr, k);
            let mut c32 = vec![0f32; mr * n];
            let mut c16 = vec![0u16; mr * n];
            for b in 0..n / 16 {
                let spec = MicroKernelSpec {
                    mr,
                    nr: 16,
                    k_pairs: k / 2,
                    k_odd: false,
                    lda_bytes: a_row_bytes(k),
                    a_fmt: AFormat::Bf16,
                    ldc_bytes: n * if out_bf16 { 2 } else { 4 },
                    n_tail: 16,
                    out_bf16,
                    opts: KernelOptions {
                        neon: NeonOptions {
                            w_int8: true,
                            w_q4: q4,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    k_dynamic: false,
                };
                let bytes = super::gemm_int8::emit(&spec).unwrap();
                let page = exec(&bytes);
                let run: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
                    unsafe { std::mem::transmute(page.as_ptr()) };
                let c = if out_bf16 {
                    unsafe { c16.as_mut_ptr().add(b * 16).cast::<u8>() }
                } else {
                    unsafe { c32.as_mut_ptr().add(b * 16).cast::<u8>() }
                };
                unsafe { run(a_q.as_ptr(), packed.as_ptr().add(b * block), c) };
            }
            for i in 0..mr {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for g in 0..k / GROUP {
                        let arow = &a_q[i * a_row_bytes(k)..];
                        let sa =
                            f32::from_le_bytes(arow[k + 8 * g..k + 8 * g + 4].try_into().unwrap());
                        let (b, lane) = (j / 16, j % 16);
                        let base = b * block + g * int8::group_bytes(q4);
                        let sw_at = base + if q4 { 256 } else { 512 } + 4 * lane;
                        let sw = f32::from_le_bytes(packed[sw_at..sw_at + 4].try_into().unwrap());
                        let mut dot = 0i32;
                        for kk in 0..32 {
                            let qw = if q4 {
                                let byte = packed[base + (kk / 8) * 64 + lane * 4 + kk % 4];
                                i32::from(if kk % 8 >= 4 { byte >> 4 } else { byte & 0xf }) - 8
                            } else {
                                i32::from(
                                    (packed[base + (kk / 4) * 64 + lane * 4 + kk % 4] ^ 0x80) as i8,
                                )
                            };
                            dot += qw * i32::from(arow[g * 32 + kk] as i8);
                        }
                        acc += f64::from(sa) * f64::from(sw) * f64::from(dot);
                    }
                    let got = if out_bf16 {
                        f64::from(f(c16[i * n + j]))
                    } else {
                        f64::from(c32[i * n + j])
                    };
                    let tol = if out_bf16 { 8e-3 } else { 1e-4 };
                    assert!(
                        (got - acc).abs() <= tol * acc.abs().max(1.0),
                        "q4 {q4} mr {mr} bf16 {out_bf16} C[{i}][{j}] = {got}, reference {acc}"
                    );
                }
            }
        }
    }
}

#[test]
fn int8_gemm_matches_reference() {
    gemm_check(false);
}

#[test]
fn q4_gemm_matches_reference() {
    gemm_check(true);
}

fn gemm_check_sse42(q4: bool) {
    if !crate::core::isa::feature("sse4.2") {
        return;
    }
    let (n, k) = (48usize, 96usize);
    let mut seed = if q4 { 5 } else { 11 };
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = if q4 {
        int8::pack_q4(&w, n, k)
    } else {
        int8::pack_int8(&w, n, k)
    };
    let block = k / GROUP * int8::group_bytes(q4);
    // SSE4.2 kernel is one row only (sixteen xmm would be needed for two).
    let mr = 1usize;
    for out_bf16 in [false, true] {
        let a_src: Vec<u16> = (0..mr * k).map(|_| bf16(rng(&mut seed) * 3.0)).collect();
        let a_q = quantize(&a_src, mr, k);
        let mut c32 = vec![0f32; mr * n];
        let mut c16 = vec![0u16; mr * n];
        for b in 0..n / 16 {
            let spec = MicroKernelSpec {
                mr,
                nr: 16,
                k_pairs: k / 2,
                k_odd: false,
                lda_bytes: a_row_bytes(k),
                a_fmt: AFormat::Bf16,
                ldc_bytes: n * if out_bf16 { 2 } else { 4 },
                n_tail: 16,
                out_bf16,
                opts: KernelOptions {
                    neon: NeonOptions {
                        w_int8: true,
                        w_q4: q4,
                        w_sse42: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                k_dynamic: false,
            };
            let bytes = super::gemm_int8_sse42::emit(&spec).unwrap();
            let page = exec(&bytes);
            let run: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
                unsafe { std::mem::transmute(page.as_ptr()) };
            let c = if out_bf16 {
                unsafe { c16.as_mut_ptr().add(b * 16).cast::<u8>() }
            } else {
                unsafe { c32.as_mut_ptr().add(b * 16).cast::<u8>() }
            };
            unsafe { run(a_q.as_ptr(), packed.as_ptr().add(b * block), c) };
        }
        for i in 0..mr {
            for j in 0..n {
                let mut acc = 0.0f64;
                for g in 0..k / GROUP {
                    let arow = &a_q[i * a_row_bytes(k)..];
                    let sa = f32::from_le_bytes(arow[k + 8 * g..k + 8 * g + 4].try_into().unwrap());
                    let (b, lane) = (j / 16, j % 16);
                    let base = b * block + g * int8::group_bytes(q4);
                    let sw_at = base + if q4 { 256 } else { 512 } + 4 * lane;
                    let sw = f32::from_le_bytes(packed[sw_at..sw_at + 4].try_into().unwrap());
                    let mut dot = 0i32;
                    for kk in 0..32 {
                        let qw = if q4 {
                            let byte = packed[base + (kk / 8) * 64 + lane * 4 + kk % 4];
                            i32::from(if kk % 8 >= 4 { byte >> 4 } else { byte & 0xf }) - 8
                        } else {
                            i32::from(
                                (packed[base + (kk / 4) * 64 + lane * 4 + kk % 4] ^ 0x80) as i8,
                            )
                        };
                        dot += qw * i32::from(arow[g * 32 + kk] as i8);
                    }
                    acc += f64::from(sa) * f64::from(sw) * f64::from(dot);
                }
                let got = if out_bf16 {
                    f64::from(f(c16[i * n + j]))
                } else {
                    f64::from(c32[i * n + j])
                };
                let tol = if out_bf16 { 8e-3 } else { 1e-4 };
                assert!(
                    (got - acc).abs() <= tol * acc.abs().max(1.0),
                    "sse42 q4 {q4} bf16 {out_bf16} C[{i}][{j}] = {got}, reference {acc}"
                );
            }
        }
    }
}

/// The SSE4.2 Q4 kernel (`pmaddubsw`/`pmaddwd` emulation of `vpdpbusd`) matches the scalar
/// reference. Only Q4: with int8 weights `pmaddubsw` saturates its 16-bit intermediate, so that
/// path is approximate.
#[test]
fn q4_gemm_sse42_matches_reference() {
    gemm_check_sse42(true);
}

#[cfg(test)]
type EmitFn = fn(&MicroKernelSpec) -> crate::core::error::Result<Vec<u8>>;

/// Time `reps` full (1×n) Q4 decode projections of the kernel emitted by `emit`, min of 5 trials.
#[cfg(test)]
fn time_q4(emit: EmitFn, n: usize, k: usize, reps: usize) -> std::time::Duration {
    let mut seed = 1u64;
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = int8::pack_q4(&w, n, k);
    let block = k / GROUP * int8::group_bytes(true);
    let a_src: Vec<u16> = (0..k).map(|_| bf16(rng(&mut seed) * 3.0)).collect();
    let a_q = quantize(&a_src, 1, k);
    let mut c = vec![0f32; n];
    let spec = MicroKernelSpec {
        mr: 1,
        nr: 16,
        k_pairs: k / 2,
        k_odd: false,
        lda_bytes: a_row_bytes(k),
        a_fmt: AFormat::Bf16,
        ldc_bytes: n * 4,
        n_tail: 16,
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
    let bytes = emit(&spec).unwrap();
    let page = exec(&bytes);
    let run: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let blocks = n / 16;
    let project = |c: &mut [f32]| {
        for b in 0..blocks {
            unsafe {
                run(
                    a_q.as_ptr(),
                    packed.as_ptr().add(b * block),
                    c.as_mut_ptr().add(b * 16).cast::<u8>(),
                )
            };
        }
    };
    for _ in 0..200 {
        project(&mut c);
    }
    let mut best = std::time::Duration::from_secs(3600);
    for _ in 0..5 {
        let t = std::time::Instant::now();
        for _ in 0..reps {
            project(&mut c);
        }
        best = best.min(t.elapsed());
    }
    std::hint::black_box(&c);
    best
}

/// `cargo test --release --lib bench_sse42_vs_vnni_q4 -- --ignored --nocapture`: the cost of
/// emulating `vpdpbusd` in SSE4.2. Runs the SSE4.2 Q4 kernel always, the AVX-VNNI one only where
/// the ISA has it (same machine → clean ratio).
#[test]
#[ignore = "a speed measurement, run by hand"]
fn bench_sse42_vs_vnni_q4() {
    let vnni = crate::core::isa::feature("avxvnni");
    println!("\n== Q4 GEMM micro-kernel: SSE4.2 (emulated) vs AVX-VNNI (vpdpbusd) ==");
    println!("AVX-VNNI available on this machine: {vnni}");
    for (label, n, k, reps) in [
        (
            "hot (weights L1-resident)",
            16usize,
            2048usize,
            200_000usize,
        ),
        ("streaming (10 MB weights)", 4096, 4096, 300),
    ] {
        let macs = (n as f64) * (k as f64) * (reps as f64);
        let sse = time_q4(super::gemm_int8_sse42::emit, n, k, reps);
        let sse_g = macs / sse.as_secs_f64() / 1e9;
        print!("[{label}] n={n} k={k}: SSE4.2 {sse_g:7.2} GMAC/s");
        if vnni {
            let v = time_q4(super::gemm_int8::emit, n, k, reps);
            let v_g = macs / v.as_secs_f64() / 1e9;
            println!(
                " | VNNI {v_g:7.2} GMAC/s | VNNI is {:.2}× faster",
                v_g / sse_g
            );
        } else {
            println!(" | (no VNNI here)");
        }
    }
    println!();
}

/// The `zmm` kernel against the `ymm` one, bit for bit: each row of a `zmm` tile of 1 to 8 rows
/// against the `ymm` kernel run on that row alone, Q4 and int8, f32 and bf16 out.
fn zmm_vs_ymm(q4: bool) {
    let f = crate::core::isa::feature;
    if !(f("avxvnni") && f("avx512f") && f("avx512bw") && f("avx512vnni")) {
        return;
    }
    let (n, k) = (64usize, 1024usize);
    let mut seed = if q4 { 17 } else { 23 };
    let w: Vec<u16> = (0..n * k)
        .map(|i| bf16(rng(&mut seed) * if i % 97 == 0 { 40.0 } else { 1.0 }))
        .collect();
    let packed = if q4 {
        int8::pack_q4(&w, n, k)
    } else {
        int8::pack_int8(&w, n, k)
    };
    let block = k / GROUP * int8::group_bytes(q4);
    for mr in 1..=super::gemm_int8::ZMM_ROWS {
        for out_bf16 in [false, true] {
            let a_src: Vec<u16> = (0..mr * k)
                .map(|i| bf16(rng(&mut seed) * if i % 53 == 0 { 30.0 } else { 2.0 }))
                .collect();
            let a_q = quantize(&a_src, mr, k);
            let es = if out_bf16 { 2 } else { 4 };
            let spec = |rows: usize| MicroKernelSpec {
                mr: rows,
                nr: 16,
                k_pairs: k / 2,
                k_odd: false,
                lda_bytes: a_row_bytes(k),
                a_fmt: AFormat::Bf16,
                ldc_bytes: n * es,
                n_tail: 16,
                out_bf16,
                opts: KernelOptions {
                    neon: NeonOptions {
                        w_int8: true,
                        w_q4: q4,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                k_dynamic: false,
            };
            let run = |bytes: &[u8], a: *const u8, c: &mut [u8]| {
                let page = exec(bytes);
                let call: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
                    unsafe { std::mem::transmute(page.as_ptr()) };
                for b in 0..n / 16 {
                    unsafe {
                        call(
                            a,
                            packed.as_ptr().add(b * block),
                            c.as_mut_ptr().add(b * 16 * es),
                        )
                    };
                }
            };
            let mut zc = vec![0u8; mr * n * es];
            run(
                &super::gemm_int8::emit_with(&spec(mr), true).unwrap(),
                a_q.as_ptr(),
                &mut zc,
            );
            let one = super::gemm_int8::emit_with(&spec(1), false).unwrap();
            for i in 0..mr {
                let mut yc = vec![0u8; n * es];
                run(
                    &one,
                    unsafe { a_q.as_ptr().add(i * a_row_bytes(k)) },
                    &mut yc,
                );
                assert_eq!(
                    &zc[i * n * es..(i + 1) * n * es],
                    &yc[..],
                    "q4 {q4} mr {mr} bf16 {out_bf16} row {i}"
                );
            }
        }
    }
}

#[test]
fn q4_zmm_matches_ymm() {
    zmm_vs_ymm(true);
}

#[test]
fn int8_zmm_matches_ymm() {
    zmm_vs_ymm(false);
}

/// `cargo test --release --lib q4_zmm_speed -- --ignored --nocapture`: the `zmm` kernel against the
/// `ymm` one, eight rows (four tiles of two on `ymm`), weights L1-resident.
#[test]
#[ignore = "a speed measurement, run by hand"]
fn q4_zmm_speed() {
    if !crate::core::isa::feature("avx512vnni") {
        return;
    }
    let (n, k, reps) = (16usize, 1024usize, 8_000usize);
    let mut seed = 3u64;
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = int8::pack_q4(&w, n, k);
    let a_src: Vec<u16> = (0..120 * k).map(|_| bf16(rng(&mut seed))).collect();
    let a_q = quantize(&a_src, 120, k);
    let mut c = vec![0u16; 120 * n];
    for (label, zmm, mr) in [
        ("ymm 2", false, 2usize),
        ("zmm 4", true, 4),
        ("zmm 6", true, 6),
        ("zmm 8", true, 8),
        ("zmm 10", true, 10),
    ] {
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
        let bytes = super::gemm_int8::emit_with(&spec, zmm).unwrap();
        let page = exec(&bytes);
        let call: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
            unsafe { std::mem::transmute(page.as_ptr()) };
        let mut best = std::time::Duration::from_secs(3600);
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                for r in (0..120).step_by(mr) {
                    unsafe {
                        call(
                            a_q.as_ptr().add(r * a_row_bytes(k)),
                            packed.as_ptr(),
                            c.as_mut_ptr().add(r * n).cast(),
                        )
                    };
                }
            }
            best = best.min(t.elapsed());
        }
        let macs = (120 * n * k * reps) as f64;
        println!(
            "{label} rows: {:.1} GMAC/s",
            macs / best.as_secs_f64() / 1e9
        );
    }
}

/// One weight block after another through the kernel `bytes`: the `mr` rows of `a_q` against the
/// `n` columns.
fn run_blocks(
    bytes: &[u8],
    a_q: &[u8],
    packed: &[u8],
    n: usize,
    block: usize,
    mr: usize,
    es: usize,
) -> Vec<u8> {
    let page = exec(bytes);
    let call: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let mut c = vec![0u8; mr * n * es];
    for b in 0..n / 16 {
        unsafe {
            call(
                a_q.as_ptr(),
                packed.as_ptr().add(b * block),
                c.as_mut_ptr().add(b * 16 * es),
            )
        };
    }
    c
}

/// The bits the `ymm` kernel must give, in scalars: per group the exact integer dot product, `cvt`
/// to f32, then `acc = fma(dot, scale_w · scale_a, acc)` from the first group on; bf16 rounded to
/// nearest even.
fn ymm_reference(
    a_q: &[u8],
    packed: &[u8],
    n: usize,
    k: usize,
    mr: usize,
    q4: bool,
    out_bf16: bool,
) -> Vec<u8> {
    let block = k / GROUP * int8::group_bytes(q4);
    let mut out = Vec::new();
    for i in 0..mr {
        let arow = &a_q[i * a_row_bytes(k)..];
        for j in 0..n {
            let mut acc = 0.0f32;
            for g in 0..k / GROUP {
                let sa = f32::from_le_bytes(arow[k + 8 * g..k + 8 * g + 4].try_into().unwrap());
                let (b, lane) = (j / 16, j % 16);
                let base = b * block + g * int8::group_bytes(q4);
                let sw_at = base + if q4 { 256 } else { 512 } + 4 * lane;
                let sw = f32::from_le_bytes(packed[sw_at..sw_at + 4].try_into().unwrap());
                let mut dot = 0i32;
                for kk in 0..32 {
                    let qw = if q4 {
                        let byte = packed[base + (kk / 8) * 64 + lane * 4 + kk % 4];
                        i32::from(if kk % 8 >= 4 { byte >> 4 } else { byte & 0xf }) - 8
                    } else {
                        i32::from((packed[base + (kk / 4) * 64 + lane * 4 + kk % 4] ^ 0x80) as i8)
                    };
                    dot += qw * i32::from(arow[g * 32 + kk] as i8);
                }
                acc = (dot as f32).mul_add(sw * sa, acc);
            }
            if out_bf16 {
                let x = acc.to_bits();
                let r = (x.wrapping_add((x >> 16) & 1).wrapping_add(0x7fff) >> 16) as u16;
                out.extend_from_slice(&r.to_le_bytes());
            } else {
                out.extend_from_slice(&acc.to_le_bytes());
            }
        }
    }
    out
}

/// The `ymm` kernel's dot products this machine can run: `vpdpbusd` with AVX-VNNI, the AVX2 forms
/// always (the engine's floor). int8 only `vpdpbusd`: `vpmaddubsw` saturates there.
fn ymm_dots(q4: bool) -> Vec<super::gemm_int8::YmmDot> {
    use super::gemm_int8::YmmDot;
    let mut dots = Vec::new();
    if crate::core::isa::feature("avxvnni") {
        dots.push(YmmDot::Vnni);
    }
    if q4 && crate::core::isa::feature("avx2") {
        dots.extend([YmmDot::Avx2Words, YmmDot::Avx2]);
    }
    dots
}

/// Every `ymm` form of the kernel, bit for bit against [`ymm_reference`] (so the AVX2 forms give
/// `vpdpbusd`'s bits): `K` from one group to 4 096, one and two rows, f32 and bf16 out; then the
/// extremes of Q4 (every code 15 or 0, activations ±127: a 16-bit lane of `Avx2Words` at
/// `±16·15·127`).
fn ymm_bits(q4: bool) {
    let dots = ymm_dots(q4);
    if dots.is_empty() {
        return;
    }
    let n = 48usize;
    let mut seed = if q4 { 31 } else { 37 };
    let mut cases = 0;
    for k in [32usize, 64, 96, 160, 1024, 4096] {
        let w: Vec<u16> = (0..n * k)
            .map(|i| bf16(rng(&mut seed) * if i % 89 == 0 { 25.0 } else { 1.0 }))
            .collect();
        let mut packed = if q4 {
            int8::pack_q4(&w, n, k)
        } else {
            int8::pack_int8(&w, n, k)
        };
        let block = k / GROUP * int8::group_bytes(q4);
        for extreme in [false, true] {
            if extreme {
                if !q4 {
                    continue;
                }
                for b in 0..n / 16 {
                    for g in 0..k / GROUP {
                        let base = b * block + g * int8::group_bytes(q4);
                        let fill = if (b + g) % 3 == 2 { 0x00 } else { 0xff };
                        packed[base..base + 256].fill(fill);
                    }
                }
            }
            for mr in 1..=super::gemm_int8::MAX_ROWS {
                for out_bf16 in [false, true] {
                    let a_src: Vec<u16> = if extreme {
                        (0..mr * k)
                            .map(|i| bf16(if (i / k + i / 64) % 2 == 0 { -1.0 } else { 1.0 }))
                            .collect()
                    } else {
                        (0..mr * k)
                            .map(|i| bf16(rng(&mut seed) * if i % 53 == 0 { 30.0 } else { 2.0 }))
                            .collect()
                    };
                    let a_q = quantize(&a_src, mr, k);
                    let es = if out_bf16 { 2 } else { 4 };
                    let spec = MicroKernelSpec {
                        mr,
                        nr: 16,
                        k_pairs: k / 2,
                        k_odd: false,
                        lda_bytes: a_row_bytes(k),
                        a_fmt: AFormat::Bf16,
                        ldc_bytes: n * es,
                        n_tail: 16,
                        out_bf16,
                        opts: KernelOptions {
                            neon: NeonOptions {
                                w_int8: true,
                                w_q4: q4,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                        k_dynamic: false,
                    };
                    let want = ymm_reference(&a_q, &packed, n, k, mr, q4, out_bf16);
                    for &dot in &dots {
                        let got = run_blocks(
                            &super::gemm_int8::emit_ymm(&spec, dot).unwrap(),
                            &a_q,
                            &packed,
                            n,
                            block,
                            mr,
                            es,
                        );
                        for (e, (g, r)) in got.chunks(es).zip(want.chunks(es)).enumerate() {
                            assert_eq!(g, r, "{dot:?} q4 {q4} k {k} extreme {extreme} mr {mr} bf16 {out_bf16} C[{}][{}]", e / n, e % n);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("ymm_bits q4 {q4}: {cases} kernels bit for bit ({dots:?})");
}

#[test]
fn q4_ymm_avx2_bits() {
    ymm_bits(true);
}

#[test]
fn int8_ymm_vnni_bits() {
    ymm_bits(false);
}

/// `cargo test --release --lib q4_ymm_dot_speed -- --ignored --nocapture`: the `ymm` kernel's dot
/// products this machine runs, one and two rows, weights L1-resident (sixteen columns, `K` 1 024,
/// 120 rows) and streaming (4 096 × 4 096, one row: a decode projection).
#[test]
#[ignore = "a speed measurement, run by hand"]
fn q4_ymm_dot_speed() {
    let (n, k, reps) = (16usize, 1024usize, 4_000usize);
    let mut seed = 3u64;
    let w: Vec<u16> = (0..n * k).map(|_| bf16(rng(&mut seed))).collect();
    let packed = int8::pack_q4(&w, n, k);
    let a_src: Vec<u16> = (0..120 * k).map(|_| bf16(rng(&mut seed))).collect();
    let a_q = quantize(&a_src, 120, k);
    let mut c = vec![0u16; 120 * n];
    for dot in ymm_dots(true) {
        for mr in [1usize, 2] {
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
            let bytes = super::gemm_int8::emit_ymm(&spec, dot).unwrap();
            let page = exec(&bytes);
            let call: unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8) =
                unsafe { std::mem::transmute(page.as_ptr()) };
            let mut best = std::time::Duration::from_secs(3600);
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    for r in (0..120).step_by(mr) {
                        unsafe {
                            call(
                                a_q.as_ptr().add(r * a_row_bytes(k)),
                                packed.as_ptr(),
                                c.as_mut_ptr().add(r * n).cast(),
                            )
                        };
                    }
                }
                best = best.min(t.elapsed());
            }
            let macs = (120 * n * k * reps) as f64;
            println!(
                "{dot:?} {mr} rows, hot: {:.1} GMAC/s",
                macs / best.as_secs_f64() / 1e9
            );
        }
    }
    let sse = time_q4(super::gemm_int8_sse42::emit, 4096, 4096, 300);
    println!(
        "SSE4.2 1 row, streaming 4096x4096: {:.2} GMAC/s",
        4096.0 * 4096.0 * 300.0 / sse.as_secs_f64() / 1e9
    );
    let ymm = time_q4(super::gemm_int8::emit, 4096, 4096, 300);
    println!(
        "gemm_int8 ({}) 1 row, streaming 4096x4096: {:.2} GMAC/s",
        super::gemm_int8::name(true),
        4096.0 * 4096.0 * 300.0 / ymm.as_secs_f64() / 1e9
    );
}
