//! The AMX packing node and GEMM kernel (`gemm_amx.rs`, `execution/bf16.rs`) against a scalar
//! reference and against the vector kernels whose bits they claim: the bf16 `bfmlal` kernel of
//! `gemm.rs`, the f16 `widen` kernel of `gemm_f16.rs` (`--f16 --f16-flush 0`). The reference is one
//! `f32::mul_add` per `k`, in order. Skipped off macOS, and the bf16 tests on a core without
//! FEAT_BF16 (the M1), whose AMX has no bf16.
use crate::jit::arch::aarch64::{gemm, gemm_amx, gemm_f16};
use crate::jit::emit::{
    AFormat, AmxF16A, KernelOptions, MicroKernelSpec, NeonMac, NeonOptions, AMX_ROWS,
};
use crate::jit::execution::abi::PackCall;

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

fn f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn f16(b: u16) -> u16 {
    half::f16::from_f32(f(b)).to_bits()
}

fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}

/// Values over several decades, signs mixed, a few zeros: in the f16 range and exact there (|x| ≥
/// 2^-14).
fn values(n: usize, seed: &mut u64, scale: f32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let r = rng(seed);
            let mag = 10f32.powf(3.0 * rng(seed).abs() - 2.0) * scale;
            if i % 37 == 0 {
                0
            } else {
                bf16(r.signum() * mag.max(1.0e-4))
            }
        })
        .collect()
}

/// The weights `[n][k]` in R16: bf16 pairs (lane `2n + t`) or, for f16, k-major pairs.
fn pack_w(w: &[u16], n: usize, k: usize, as_f16: bool) -> Vec<u16> {
    let blocks = n.div_ceil(16);
    let mut out = vec![0u16; blocks * k * 16];
    for b in 0..blocks {
        for kp in 0..k / 2 {
            let base = (b * (k / 2) + kp) * 32;
            for lane in 0..16.min(n - b * 16) {
                let row = b * 16 + lane;
                let (v0, v1) = (w[row * k + 2 * kp], w[row * k + 2 * kp + 1]);
                if as_f16 {
                    out[base + lane] = f16(v0);
                    out[base + 16 + lane] = f16(v1);
                } else {
                    out[base + 2 * lane] = v0;
                    out[base + 2 * lane + 1] = v1;
                }
            }
        }
    }
    out
}

/// The options of the AMX kernels: bf16 weights, or f16 with the activations in the form `amx_a`.
fn neon_of(as_f16: bool, amx_a: AmxF16A) -> NeonOptions {
    NeonOptions {
        w_f16: as_f16,
        amx_a,
        ..Default::default()
    }
}

/// Runs the packing node over `rows` rows of `a` (`k` per row): the packed bytes.
fn pack_a(a: &[u16], rows: usize, k: usize, neon: NeonOptions) -> Vec<u8> {
    let mut dest = vec![0u8; crate::jit::emit::packed_a_amx_bytes(rows, k, neon)];
    let kernel = super::execution::bf16::pack_amx(k, neon).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "C" fn(*const PackCall, usize) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let n = rows;
    let call = PackCall {
        source: a.as_ptr(),
        dest: dest.as_mut_ptr().cast(),
        rows: &n,
    };
    for block in 0..rows.div_ceil(AMX_ROWS) {
        unsafe { run(&call, block) };
    }
    dest
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn spec(
    mr: usize,
    k: usize,
    n: usize,
    block: usize,
    out_bf16: bool,
    opts: KernelOptions,
) -> MicroKernelSpec {
    let ldc = n.div_ceil(16) * 16;
    MicroKernelSpec {
        mr,
        nr: 16,
        k_pairs: k / 2,
        k_odd: false,
        lda_bytes: k * 2,
        a_fmt: AFormat::Bf16,
        ldc_bytes: ldc * if out_bf16 { 2 } else { 4 },
        n_tail: if (block + 1) * 16 > n {
            n - block * 16
        } else {
            16
        },
        out_bf16,
        opts,
        k_dynamic: false,
    }
}

/// `C` (f32, `rows × ldc`) from the AMX kernel, one call per block of 32 rows and of 16 columns.
fn amx_gemm(
    a: &[u16],
    w_packed: &[u16],
    rows: usize,
    k: usize,
    n: usize,
    neon: NeonOptions,
) -> Vec<f32> {
    let packed = pack_a(a, rows, k, neon);
    let block_bytes = AMX_ROWS * crate::jit::emit::amx_row_bytes(k, neon);
    let ldc = n.div_ceil(16) * 16;
    let mut c = vec![f32::NAN; rows * ldc];
    let opts = KernelOptions {
        matrix: true,
        amx: true,
        neon,
        ..Default::default()
    };
    for rb in 0..rows.div_ceil(AMX_ROWS) {
        let mr = AMX_ROWS.min(rows - rb * AMX_ROWS);
        for block in 0..n.div_ceil(16) {
            let bytes = gemm_amx::emit(&spec(mr, k, n, block, false, opts)).unwrap();
            let page = exec(&bytes);
            let run: unsafe extern "C" fn(*const u8, *const u16, *mut f32) =
                unsafe { std::mem::transmute(page.as_ptr()) };
            unsafe {
                run(
                    packed.as_ptr().add(rb * block_bytes),
                    w_packed.as_ptr().add(block * k * 16),
                    c.as_mut_ptr().add(rb * AMX_ROWS * ldc + block * 16),
                )
            };
        }
    }
    c
}

/// `C` from the vector kernel of these options, tiles of up to four rows.
fn vector_gemm(
    a: &[u16],
    w_packed: &[u16],
    rows: usize,
    k: usize,
    n: usize,
    neon: NeonOptions,
) -> Vec<f32> {
    let ldc = n.div_ceil(16) * 16;
    let mut c = vec![f32::NAN; rows * ldc];
    let opts = KernelOptions {
        neon,
        ..Default::default()
    };
    for r0 in (0..rows).step_by(4) {
        let mr = 4.min(rows - r0);
        for block in 0..n.div_ceil(16) {
            let s = spec(mr, k, n, block, false, opts);
            let bytes = if neon.w_f16 {
                gemm_f16::emit(&s)
            } else {
                gemm::emit(&s)
            }
            .unwrap();
            let page = exec(&bytes);
            let run: unsafe extern "C" fn(*const u16, *const u16, *mut f32) =
                unsafe { std::mem::transmute(page.as_ptr()) };
            unsafe {
                run(
                    a.as_ptr().add(r0 * k),
                    w_packed.as_ptr().add(block * k * 16),
                    c.as_mut_ptr().add(r0 * ldc + block * 16),
                )
            };
        }
    }
    c
}

/// One fused multiply-add per `k`, in order, from +0.
fn reference(a: &[u16], w: &[u16], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let ldc = n.div_ceil(16) * 16;
    let mut c = vec![f32::NAN; rows * ldc];
    for i in 0..rows {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = f(a[i * k + kk]).mul_add(f(w[j * k + kk]), acc);
            }
            c[i * ldc + j] = acc;
        }
    }
    c
}

fn assert_bits(got: &[f32], want: &[f32], rows: usize, n: usize, what: &str) {
    let ldc = n.div_ceil(16) * 16;
    for i in 0..rows {
        for j in 0..n {
            let (g, w) = (got[i * ldc + j], want[i * ldc + j]);
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: C[{i}][{j}] = {g:e}, expected {w:e}"
            );
        }
    }
}

/// Activations far outside f16: tiny (down to the bf16 subnormals) and huge, signs mixed, among
/// ordinary ones.
fn wild(n: usize, seed: &mut u64) -> Vec<u16> {
    const ODD: [f32; 12] = [
        3.0e-6, -7.1e-7, 1.0e-10, -2.5e-20, 1.0e-38, -3.0e-40, 9.2e-41, 7.0e4, -1.3e5, 2.0e10,
        -6.0e20, 1.0e30,
    ];
    (0..n)
        .map(|i| {
            let r = rng(seed);
            if i % 5 == 0 {
                bf16(ODD[(i / 5) % ODD.len()] * (1.0 + 0.3 * r))
            } else {
                bf16(r * 2.0)
            }
        })
        .collect()
}

/// f16 weights with subnormals (from 2^-24) and large values among ordinary ones.
fn wild_w(n: usize, seed: &mut u64) -> Vec<u16> {
    const ODD: [f32; 6] = [6.0e-8, -3.1e-6, 4.0e-5, 6.0e4, -2.0e4, 1.0e-7];
    (0..n)
        .map(|i| {
            let r = rng(seed);
            if i % 7 == 0 {
                bf16(half::f16::from_f32(ODD[(i / 7) % ODD.len()] * (1.0 + 0.3 * r)).to_f32())
            } else {
                bf16(r * 0.05)
            }
        })
        .collect()
}

#[test]
fn pack_converts_and_transposes() {
    if !gemm_amx::available() {
        return;
    }
    let (rows, k) = (45usize, 64usize);
    let mut seed = 3u64;
    // Tiny and large values too: the f16 conversion must round as `half` does (nearest even).
    let mut a = values(rows * k, &mut seed, 1.0);
    a[5] = bf16(1.0e-7);
    a[6] = bf16(-3.3e-6);
    a[7] = bf16(60000.0);
    // Those two cannot come back from f16: under `Mixed` block 0 is f32; block 1 is f16, then f32
    // again with one more such value.
    let mut b = a.clone();
    b[40 * k + 3] = bf16(1.0e-30);
    let forms = [
        (false, AmxF16A::F32),
        (true, AmxF16A::F16),
        (true, AmxF16A::F32),
        (true, AmxF16A::Mixed),
    ];
    for (a, (as_f16, amx_a)) in [&a, &b].into_iter().flat_map(|a| forms.map(|f| (a, f))) {
        let neon = neon_of(as_f16, amx_a);
        let p = pack_a(a, rows, k, neon);
        let block_bytes = AMX_ROWS * crate::jit::emit::amx_row_bytes(k, neon);
        for blk in 0..rows.div_ceil(AMX_ROWS) {
            let (mut base, mut wide) = (blk * block_bytes, as_f16 && amx_a == AmxF16A::F32);
            if as_f16 && amx_a == AmxF16A::Mixed {
                let rows_of = (blk * AMX_ROWS..rows.min((blk + 1) * AMX_ROWS))
                    .flat_map(|r| &a[r * k..(r + 1) * k]);
                let back =
                    |v: &u16| half::f16::from_bits(f16(*v)).to_f32().to_bits() == f(*v).to_bits();
                wide = !rows_of.clone().all(back);
                assert_eq!(
                    u32_at(&p, base),
                    u32::from(wide),
                    "Mixed: the header of block {blk}"
                );
                base += 64;
            }
            for kk in 0..k {
                for r in 0..AMX_ROWS {
                    let src = (blk * AMX_ROWS + r).min(rows - 1);
                    let v = a[src * k + kk];
                    let what = format!("f16 {as_f16} {amx_a:?} block {blk} K {kk} row {r}");
                    if wide {
                        assert_eq!(
                            u32_at(&p, base + (kk * AMX_ROWS + r) * 4),
                            f(v).to_bits(),
                            "{what}"
                        );
                    } else {
                        let want = if as_f16 { f16(v) } else { v };
                        assert_eq!(u16_at(&p, base + (kk * AMX_ROWS + r) * 2), want, "{what}");
                    }
                }
            }
        }
    }
}

#[test]
fn bf16_matches_bfmlal() {
    if !gemm_amx::serves(false) {
        return;
    }
    let (n, k) = (40usize, 136usize);
    let mut seed = 11u64;
    let w = values(n * k, &mut seed, 0.05);
    let wp = pack_w(&w, n, k, false);
    for rows in [32usize, 7, 45] {
        let a = values(rows * k, &mut seed, 2.0);
        let want = reference(&a, &w, rows, k, n);
        assert_bits(
            &amx_gemm(&a, &wp, rows, k, n, neon_of(false, AmxF16A::F32)),
            &want,
            rows,
            n,
            "amx bf16 vs reference",
        );
        let neon = NeonOptions {
            mac: NeonMac::BfMlal,
            ..Default::default()
        };
        assert_bits(
            &vector_gemm(&a, &wp, rows, k, n, neon),
            &want,
            rows,
            n,
            "bfmlal vs reference",
        );
    }
}

/// In f16's range, every form of the activations gives the bits of `widen`.
#[test]
fn f16_matches_widen() {
    if !gemm_amx::serves(true) {
        return;
    }
    let (n, k) = (40usize, 136usize);
    let mut seed = 13u64;
    let w = values(n * k, &mut seed, 0.05);
    let wp = pack_w(&w, n, k, true);
    for rows in [32usize, 7, 45] {
        let a = values(rows * k, &mut seed, 2.0);
        let want = reference(&a, &w, rows, k, n);
        for amx_a in [AmxF16A::F32, AmxF16A::Mixed, AmxF16A::F16] {
            assert_bits(
                &amx_gemm(&a, &wp, rows, k, n, neon_of(true, amx_a)),
                &want,
                rows,
                n,
                &format!("amx f16 {amx_a:?} vs reference"),
            );
        }
        let neon = NeonOptions {
            w_f16: true,
            f16_flush: 0,
            ..Default::default()
        };
        assert_bits(
            &vector_gemm(&a, &wp, rows, k, n, neon),
            &want,
            rows,
            n,
            "f16 widen vs reference",
        );
    }
}

/// Outside f16's range — activations down to the bf16 subnormals and up to 1e30, f16 weights from
/// the subnormals to 6e4 — `F32` and `Mixed` keep the bits of `widen`; `F16` does not (the limit it
/// documents).
#[test]
fn f16_exact_outside_f16_range() {
    if !gemm_amx::serves(true) {
        return;
    }
    let (n, k) = (40usize, 136usize);
    let mut seed = 17u64;
    let w = wild_w(n * k, &mut seed);
    let wp = pack_w(&w, n, k, true);
    for rows in [32usize, 7, 45, 64, 33] {
        // Rows 0..31 in range when there are more than 32 (a block of each kind under `Mixed`);
        // with 33, every activation is tiny (1e-39 to 1e-36): the products and the sums
        // fall among the f32 subnormals, where a product rounded before its addition would
        // change the bits.
        let mut a = wild(rows * k, &mut seed);
        if rows == 33 {
            a = (0..rows * k)
                .map(|_| {
                    bf16(rng(&mut seed).signum() * 10f32.powf(-39.0 + 3.0 * rng(&mut seed).abs()))
                })
                .collect();
        } else if rows > 32 {
            a[..32 * k].copy_from_slice(&values(32 * k, &mut seed, 2.0));
        }
        let want = reference(&a, &w, rows, k, n);
        let neon = NeonOptions {
            w_f16: true,
            f16_flush: 0,
            ..Default::default()
        };
        assert_bits(
            &vector_gemm(&a, &wp, rows, k, n, neon),
            &want,
            rows,
            n,
            "f16 widen vs reference",
        );
        for amx_a in [AmxF16A::F32, AmxF16A::Mixed] {
            assert_bits(
                &amx_gemm(&a, &wp, rows, k, n, neon_of(true, amx_a)),
                &want,
                rows,
                n,
                &format!("amx f16 {amx_a:?} vs reference"),
            );
        }
        let fast = amx_gemm(&a, &wp, rows, k, n, neon_of(true, AmxF16A::F16));
        assert!(
            fast.iter()
                .zip(&want)
                .any(|(g, w)| g.to_bits() != w.to_bits()),
            "F16 kept every bit outside its range ({rows} rows)"
        );
    }
}

/// One fused multiply-add per `k`, not a product rounded then added: in row 0 the products 2^-126,
/// 2^-149, then 2^-150 (2^-133 · 2^-17, a bf16 subnormal by an f16 subnormal), a tie against the
/// odd sum 2^-126 + 2^-149. Fused, it rounds to even, 2^-126 + 2^-148; rounded first, 2^-150 is 0
/// and the sum stays. Row 1 is the same tie against an even sum (2^-126 + 2^-148): it stays.
#[test]
fn f16_exact_fused_at_a_subnormal_tie() {
    if !gemm_amx::serves(true) {
        return;
    }
    let (n, k, rows) = (16usize, 8usize, 2usize);
    let p2 = |e: i32| bf16(2f32.powi(e));
    let mut a = vec![0u16; rows * k];
    let mut w = vec![0u16; n * k];
    a[..3].copy_from_slice(&[p2(-126), p2(-133), p2(-133)]);
    a[k..k + 3].copy_from_slice(&[p2(-126), p2(-132), p2(-133)]);
    for j in 0..n {
        w[j * k..j * k + 3].copy_from_slice(&[p2(0), p2(-16), p2(-17)]);
    }
    let want = reference(&a, &w, rows, k, n);
    assert_eq!(
        want[0].to_bits(),
        (2f32.powi(-126) + 2f32.powi(-148)).to_bits(),
        "the reference rounds the tie to even"
    );
    assert_eq!(
        want[16].to_bits(),
        (2f32.powi(-126) + 2f32.powi(-148)).to_bits(),
        "the reference keeps the even sum"
    );
    let wp = pack_w(&w, n, k, true);
    let neon = NeonOptions {
        w_f16: true,
        f16_flush: 0,
        ..Default::default()
    };
    assert_bits(
        &vector_gemm(&a, &wp, rows, k, n, neon),
        &want,
        rows,
        n,
        "f16 widen vs reference",
    );
    for amx_a in [AmxF16A::F32, AmxF16A::Mixed] {
        assert_bits(
            &amx_gemm(&a, &wp, rows, k, n, neon_of(true, amx_a)),
            &want,
            rows,
            n,
            &format!("amx f16 {amx_a:?} vs reference"),
        );
    }
}

/// The one difference left: a NaN activation. The unit returns the default NaN (`0x7fc00000`, as
/// under `FPCR.DN`), `widen` the NaN of its input, quieted (`0x7fc10000` from the bf16 `0x7fc1` or
/// `0x7f81`); both are NaN. An infinity, and activations near the top of f32 (3e38), keep the bits
/// of `widen`.
#[test]
fn f16_exact_nan_payload() {
    if !gemm_amx::serves(true) {
        return;
    }
    let (n, k, rows) = (16usize, 8usize, 4usize);
    let mut seed = 5u64;
    let w = values(n * k, &mut seed, 0.05);
    let mut a = values(rows * k, &mut seed, 1.0);
    a[0] = 0x7fc1;
    a[k] = 0x7f81;
    a[2 * k] = 0x7f80;
    a[3 * k..3 * k + 8].fill(bf16(3.0e38));
    let wp = pack_w(&w, n, k, true);
    let neon = NeonOptions {
        w_f16: true,
        f16_flush: 0,
        ..Default::default()
    };
    let widen = vector_gemm(&a, &wp, rows, k, n, neon);
    for amx_a in [AmxF16A::F32, AmxF16A::Mixed] {
        let got = amx_gemm(&a, &wp, rows, k, n, neon_of(true, amx_a));
        for j in 0..n {
            for i in 0..2 {
                assert_eq!(
                    (got[i * 16 + j].to_bits(), widen[i * 16 + j].to_bits()),
                    (0x7fc0_0000, 0x7fc1_0000),
                    "{amx_a:?} C[{i}][{j}]"
                );
            }
        }
        assert_bits(
            &got[2 * 16..],
            &widen[2 * 16..],
            2,
            n,
            &format!("amx f16 {amx_a:?} vs widen, infinities"),
        );
    }
}
