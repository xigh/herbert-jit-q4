//! The AVX2 block attention against the AVX2 decode kernel called row by row (x86-64, ymm only: it
//! runs on an AVX-512 machine too). Bit for bit: the block takes each row through the decode
//! kernel's sequence.
use crate::jit::emit_sdpa::{SdpaHeadFn, SdpaKernelSpec, SdpaPrefillFn, SdpaPrefillSpec};
use crate::jit::emit_sdpa_avx2::{block_scores_row, emit_sdpa_avx2, emit_sdpa_prefill_avx2};
use crate::jit::ir::{ExpStrategy, SdpaForm, SdpaShape};

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}

/// Every row of a block (`rows` 1 to 8, `past` 0 to 41, heads of 64, 128 and 256, with and without
/// a sink, `q` and `out` strided as on a gated model) against the decode kernel on that row's `past
/// + i + 1` keys.
#[test]
fn sdpa_block_avx2_matches_rows() {
    let mut seed = 11u64;
    let mut checked = 0usize;
    for head_dim in [64usize, 128, 256] {
        for sink in [false, true] {
            let shape = SdpaShape { head_dim, sink };
            let decode = SdpaKernelSpec {
                shape,
                form: SdpaForm::Chained,
                exp: ExpStrategy::Inline,
                pf_dist: 4,
                qk_group: 0,
                sv_spill_all: false,
            };
            let dpage = exec(&emit_sdpa_avx2(&decode).unwrap());
            let head: SdpaHeadFn = unsafe { std::mem::transmute(dpage.as_ptr()) };
            for rows in [1usize, 2, 3, 5, 8] {
                for past in [0usize, 1, 7, 8, 16, 41] {
                    let cap = past + rows + 3;
                    let (q_row, out_row) = (2 * head_dim + 16, head_dim + 32);
                    let spec = SdpaPrefillSpec {
                        shape,
                        rows,
                        row_stride: q_row,
                        out_row_stride: out_row,
                        exp: ExpStrategy::Inline,
                        pf_dist: 0,
                        dim_blocks: 0,
                        qk_group: 0,
                        qk_pair: 1,
                        kv_capacity: cap,
                        full: false,
                        window: 0,
                        sv_tile: 0,
                    };
                    let bpage = exec(&emit_sdpa_prefill_avx2(&spec).unwrap());
                    let block: SdpaPrefillFn = unsafe { std::mem::transmute(bpage.as_ptr()) };
                    let n = past + rows;
                    let q: Vec<u16> = (0..rows * q_row)
                        .map(|_| bf16(rng(&mut seed) * 2.0))
                        .collect();
                    let k: Vec<u16> = (0..n * head_dim)
                        .map(|_| bf16(rng(&mut seed) * 2.0))
                        .collect();
                    let v: Vec<u16> = (0..n * head_dim).map(|_| bf16(rng(&mut seed))).collect();
                    let (scale, sink_logit) =
                        (1.0 / (head_dim as f32).sqrt(), rng(&mut seed) * 3.0);
                    let mut scores = vec![f32::NAN; rows * block_scores_row(&spec)];
                    let mut got = vec![0xAAAAu16; rows * out_row];
                    unsafe {
                        block(
                            q.as_ptr(),
                            k.as_ptr(),
                            v.as_ptr(),
                            scores.as_mut_ptr(),
                            got.as_mut_ptr(),
                            past as u64,
                            scale,
                            sink_logit,
                        )
                    };
                    let mut want = vec![0xAAAAu16; rows * out_row];
                    for i in 0..rows {
                        let seen = past + i + 1;
                        let mut s = vec![f32::NAN; seen.next_multiple_of(16)];
                        unsafe {
                            head(
                                q[i * q_row..].as_ptr(),
                                k.as_ptr(),
                                v.as_ptr(),
                                s.as_mut_ptr(),
                                want[i * out_row..].as_mut_ptr(),
                                seen as u64,
                                scale,
                                sink_logit,
                            )
                        };
                    }
                    assert_eq!(
                        got, want,
                        "head_dim {head_dim} sink {sink} rows {rows} past {past}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 3 * 2 * 5 * 6);
}

/// The block with a sliding window (`SdpaPrefillSpec::window`): every row against the decode kernel
/// on that row's window, `[max(0, past + i + 1 − window), past + i + 1)`. The block gets the
/// absolute `past` and the cache from position 0, and moves past the keys no row sees. Windows of
/// 1, 3 (fewer than the rows), 8, 33 (odd) and 64 (longer than every `past + rows` here: the causal
/// block's bits), `past` below, on and past `window − 1`, rows 1 to 8, heads of 64 to 256, with and
/// without a sink; the scratch is `scratch_len`, filled with NaN.
#[test]
fn sdpa_block_avx2_window_matches_rows() {
    let mut seed = 23u64;
    let mut checked = 0usize;
    for head_dim in [64usize, 128, 256] {
        for sink in [false, true] {
            let shape = SdpaShape { head_dim, sink };
            let decode = SdpaKernelSpec {
                shape,
                form: SdpaForm::Chained,
                exp: ExpStrategy::Inline,
                pf_dist: 4,
                qk_group: 0,
                sv_spill_all: false,
            };
            let dpage = exec(&emit_sdpa_avx2(&decode).unwrap());
            let head: SdpaHeadFn = unsafe { std::mem::transmute(dpage.as_ptr()) };
            for window in [1usize, 3, 8, 33, 64] {
                for rows in [1usize, 3, 5, 8] {
                    for past in [0usize, 1, 2, 7, 9, 31, 32, 33, 40, 47] {
                        let cap = past + rows + 3;
                        let (q_row, out_row) = (2 * head_dim + 16, head_dim + 32);
                        let spec = SdpaPrefillSpec {
                            shape,
                            rows,
                            row_stride: q_row,
                            out_row_stride: out_row,
                            exp: ExpStrategy::Inline,
                            pf_dist: 0,
                            dim_blocks: 0,
                            qk_group: 0,
                            qk_pair: 1,
                            kv_capacity: cap,
                            full: false,
                            window,
                            sv_tile: 0,
                        };
                        assert_eq!(
                            rows * block_scores_row(&spec),
                            spec.scratch_len(),
                            "the scratch of a windowed block"
                        );
                        let bpage = exec(&emit_sdpa_prefill_avx2(&spec).unwrap());
                        let block: SdpaPrefillFn = unsafe { std::mem::transmute(bpage.as_ptr()) };
                        let n = past + rows;
                        let q: Vec<u16> = (0..rows * q_row)
                            .map(|_| bf16(rng(&mut seed) * 2.0))
                            .collect();
                        let k: Vec<u16> = (0..n * head_dim)
                            .map(|_| bf16(rng(&mut seed) * 2.0))
                            .collect();
                        let v: Vec<u16> = (0..n * head_dim).map(|_| bf16(rng(&mut seed))).collect();
                        let (scale, sink_logit) =
                            (1.0 / (head_dim as f32).sqrt(), rng(&mut seed) * 3.0);
                        let mut scores = vec![f32::NAN; spec.scratch_len()];
                        let mut got = vec![0xAAAAu16; rows * out_row];
                        unsafe {
                            block(
                                q.as_ptr(),
                                k.as_ptr(),
                                v.as_ptr(),
                                scores.as_mut_ptr(),
                                got.as_mut_ptr(),
                                past as u64,
                                scale,
                                sink_logit,
                            )
                        };
                        let mut want = vec![0xAAAAu16; rows * out_row];
                        for i in 0..rows {
                            let end = past + i + 1;
                            let start = end.saturating_sub(window);
                            let seen = end - start;
                            let mut s = vec![f32::NAN; seen.next_multiple_of(16)];
                            unsafe {
                                head(
                                    q[i * q_row..].as_ptr(),
                                    k[start * head_dim..].as_ptr(),
                                    v[start * head_dim..].as_ptr(),
                                    s.as_mut_ptr(),
                                    want[i * out_row..].as_mut_ptr(),
                                    seen as u64,
                                    scale,
                                    sink_logit,
                                )
                            };
                        }
                        assert_eq!(got, want, "head_dim {head_dim} sink {sink} window {window} rows {rows} past {past}");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 3 * 2 * 5 * 4 * 10);
}

/// The block's scratch (`scratch_len`, NaN) then a sentinel: a kernel writing past its slot fails.
const SENTINEL: usize = 64;

fn scratch(spec: &SdpaPrefillSpec) -> Vec<f32> {
    let mut s = vec![f32::NAN; spec.scratch_len() + SENTINEL];
    s[spec.scratch_len()..].fill(f32::from_bits(0x7fc0_beef));
    s
}

fn sentinel_kept(s: &[f32], spec: &SdpaPrefillSpec) -> bool {
    s[spec.scratch_len()..]
        .iter()
        .all(|x| x.to_bits() == 0x7fc0_beef)
}

/// The block's signature with the end of the keys as a seventh integer argument, on the stack.
type BidirFn = unsafe extern "sysv64" fn(
    *const u16,
    *const u16,
    *const u16,
    *mut f32,
    *mut u16,
    u64,
    u64,
    f32,
    f32,
);

/// The bidirectional block takes the end of the keys in `rax` (the selector's convention): an
/// emitted trampoline moves the seventh argument there and jumps to the kernel.
fn trampoline(kernel: *const u8) -> memmap2::Mmap {
    use iced_x86::code_asm::*;
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rax, qword_ptr(rsp + 8)).unwrap();
    a.mov(r11, kernel as u64).unwrap();
    a.jmp(r11).unwrap();
    exec(&a.assemble(0).unwrap())
}

/// The block without the causal mask (`SdpaPrefillSpec::full`): every row sees the `n` keys of the
/// sixth argument, against the decode kernel on those `n` keys. Rows 1 to 8, `n` from 1 to 100,
/// heads of 64 to 256, with and without a sink; scratch bounded by a sentinel.
#[test]
fn sdpa_block_avx2_full_matches_rows() {
    let mut seed = 31u64;
    let mut checked = 0usize;
    for head_dim in [64usize, 128, 256] {
        for sink in [false, true] {
            let shape = SdpaShape { head_dim, sink };
            let decode = SdpaKernelSpec {
                shape,
                form: SdpaForm::Chained,
                exp: ExpStrategy::Inline,
                pf_dist: 4,
                qk_group: 0,
                sv_spill_all: false,
            };
            let dpage = exec(&emit_sdpa_avx2(&decode).unwrap());
            let head: SdpaHeadFn = unsafe { std::mem::transmute(dpage.as_ptr()) };
            for rows in [1usize, 3, 5, 8] {
                for n in [1usize, 7, 8, 9, 33, 100] {
                    let cap = n + 3;
                    let (q_row, out_row) = (2 * head_dim + 16, head_dim + 32);
                    let spec = SdpaPrefillSpec {
                        shape,
                        rows,
                        row_stride: q_row,
                        out_row_stride: out_row,
                        exp: ExpStrategy::Inline,
                        pf_dist: 0,
                        dim_blocks: 0,
                        qk_group: 0,
                        qk_pair: 1,
                        kv_capacity: cap,
                        full: true,
                        window: 0,
                        sv_tile: 0,
                    };
                    let bpage = exec(&emit_sdpa_prefill_avx2(&spec).unwrap());
                    let block: SdpaPrefillFn = unsafe { std::mem::transmute(bpage.as_ptr()) };
                    let q: Vec<u16> = (0..rows * q_row)
                        .map(|_| bf16(rng(&mut seed) * 2.0))
                        .collect();
                    let k: Vec<u16> = (0..n * head_dim)
                        .map(|_| bf16(rng(&mut seed) * 2.0))
                        .collect();
                    let v: Vec<u16> = (0..n * head_dim).map(|_| bf16(rng(&mut seed))).collect();
                    let (scale, sink_logit) =
                        (1.0 / (head_dim as f32).sqrt(), rng(&mut seed) * 3.0);
                    let mut scores = scratch(&spec);
                    let mut got = vec![0xAAAAu16; rows * out_row];
                    unsafe {
                        block(
                            q.as_ptr(),
                            k.as_ptr(),
                            v.as_ptr(),
                            scores.as_mut_ptr(),
                            got.as_mut_ptr(),
                            n as u64,
                            scale,
                            sink_logit,
                        )
                    };
                    assert!(
                        sentinel_kept(&scores, &spec),
                        "head_dim {head_dim} rows {rows} n {n}: past the scratch"
                    );
                    let mut want = vec![0xAAAAu16; rows * out_row];
                    for i in 0..rows {
                        let mut s = vec![f32::NAN; n.next_multiple_of(16)];
                        unsafe {
                            head(
                                q[i * q_row..].as_ptr(),
                                k.as_ptr(),
                                v.as_ptr(),
                                s.as_mut_ptr(),
                                want[i * out_row..].as_mut_ptr(),
                                n as u64,
                                scale,
                                sink_logit,
                            )
                        };
                    }
                    assert_eq!(
                        got, want,
                        "full: head_dim {head_dim} sink {sink} rows {rows} n {n}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 3 * 2 * 4 * 6);
}

/// The block without the causal mask under the bidirectional window (`full && window`, `|q − k| <
/// window`, EmbeddingGemma 2's sliding layers): the tile's absolute `past` in r9, the end of the
/// keys in rax, the cache from position 0; row `i` against the decode kernel on `[max(0, past + i +
/// 1 − window), min(past + i + window, end))`. Windows 1, 3, 8, 33 and 64, `past` below, on and
/// past `window − 1`, ends at the tile's last row and up to 40 keys past it, rows 1 to 8, heads of
/// 64 to 256, with and without a sink; scratch bounded by a sentinel.
#[test]
fn sdpa_block_avx2_bidir_window_matches_rows() {
    let mut seed = 37u64;
    let mut checked = 0usize;
    for head_dim in [64usize, 128, 256] {
        for sink in [false, true] {
            let shape = SdpaShape { head_dim, sink };
            let decode = SdpaKernelSpec {
                shape,
                form: SdpaForm::Chained,
                exp: ExpStrategy::Inline,
                pf_dist: 4,
                qk_group: 0,
                sv_spill_all: false,
            };
            let dpage = exec(&emit_sdpa_avx2(&decode).unwrap());
            let head: SdpaHeadFn = unsafe { std::mem::transmute(dpage.as_ptr()) };
            for window in [1usize, 3, 8, 33, 64] {
                for rows in [1usize, 3, 5, 8] {
                    for past in [0usize, 1, 7, 31, 32, 33, 47] {
                        for extra in [0usize, 1, 6, 40] {
                            let end = past + rows + extra;
                            let cap = end + 3;
                            let (q_row, out_row) = (2 * head_dim + 16, head_dim + 32);
                            let spec = SdpaPrefillSpec {
                                shape,
                                rows,
                                row_stride: q_row,
                                out_row_stride: out_row,
                                exp: ExpStrategy::Inline,
                                pf_dist: 0,
                                dim_blocks: 0,
                                qk_group: 0,
                                qk_pair: 1,
                                kv_capacity: cap,
                                full: true,
                                window,
                                sv_tile: 0,
                            };
                            let bpage = exec(&emit_sdpa_prefill_avx2(&spec).unwrap());
                            let tpage = trampoline(bpage.as_ptr());
                            let block: BidirFn = unsafe { std::mem::transmute(tpage.as_ptr()) };
                            let q: Vec<u16> = (0..rows * q_row)
                                .map(|_| bf16(rng(&mut seed) * 2.0))
                                .collect();
                            let k: Vec<u16> = (0..end * head_dim)
                                .map(|_| bf16(rng(&mut seed) * 2.0))
                                .collect();
                            let v: Vec<u16> =
                                (0..end * head_dim).map(|_| bf16(rng(&mut seed))).collect();
                            let (scale, sink_logit) =
                                (1.0 / (head_dim as f32).sqrt(), rng(&mut seed) * 3.0);
                            let mut scores = scratch(&spec);
                            let mut got = vec![0xAAAAu16; rows * out_row];
                            unsafe {
                                block(
                                    q.as_ptr(),
                                    k.as_ptr(),
                                    v.as_ptr(),
                                    scores.as_mut_ptr(),
                                    got.as_mut_ptr(),
                                    past as u64,
                                    end as u64,
                                    scale,
                                    sink_logit,
                                )
                            };
                            let case = format!("bidir: head_dim {head_dim} sink {sink} window {window} rows {rows} past {past} end {end}");
                            assert!(sentinel_kept(&scores, &spec), "{case}: past the scratch");
                            let mut want = vec![0xAAAAu16; rows * out_row];
                            for i in 0..rows {
                                let start = (past + i + 1).saturating_sub(window);
                                let stop = (past + i + window).min(end);
                                let seen = stop - start;
                                let mut s = vec![f32::NAN; seen.next_multiple_of(16)];
                                unsafe {
                                    head(
                                        q[i * q_row..].as_ptr(),
                                        k[start * head_dim..].as_ptr(),
                                        v[start * head_dim..].as_ptr(),
                                        s.as_mut_ptr(),
                                        want[i * out_row..].as_mut_ptr(),
                                        seen as u64,
                                        scale,
                                        sink_logit,
                                    )
                                };
                            }
                            assert_eq!(got, want, "{case}");
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 3 * 2 * 5 * 4 * 7 * 4);
}
