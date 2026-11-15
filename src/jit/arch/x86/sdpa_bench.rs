//! The AVX-512 attention kernels: the digest of their emitted bytes over a matrix of specs (a
//! change that must leave a shape's code unchanged is checked by comparing two runs of
//! [`sdpa_emitted_digest`]), and a measure of the block kernel without mask on the vision tower's
//! shape ([`bench_sdpa_block_full`]). Both `#[ignore]`: `cargo test --release --lib sdpa_bench --
//! --ignored --nocapture --test-threads=1`.
use crate::jit::emit_sdpa::{
    emit_sdpa, emit_sdpa_prefill, Pass, SdpaKernelSpec, SdpaPrefillFn, SdpaPrefillSpec,
    SDPA_PF_DIST_PREFILL,
};
use crate::jit::ir::{ExpStrategy, SdpaForm, SdpaShape};

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn shape(d: usize) -> SdpaShape {
    SdpaShape {
        head_dim: d,
        sink: false,
    }
}

/// One line per emitted spec: its key and the FNV-1a of its bytes (or `declined`).
#[test]
#[ignore = "a measurement, run by hand"]
fn sdpa_emitted_digest() {
    let mut all = 0u64;
    let mut lines = 0usize;
    for d in [64usize, 72, 80, 88, 96, 104, 128, 256] {
        for sink in [false, true] {
            let sh = SdpaShape { head_dim: d, sink };
            for form in [SdpaForm::Chained, SdpaForm::Split, SdpaForm::Fused] {
                for qk_group in [0u8, 8, 16] {
                    for spill in [false, true] {
                        let spec = SdpaKernelSpec {
                            shape: sh,
                            form,
                            qk_group,
                            sv_spill_all: spill,
                            ..SdpaKernelSpec::default()
                        };
                        let passes: Vec<Option<Pass>> = if form == SdpaForm::Split {
                            vec![Some(Pass::Qk), Some(Pass::Softmax), Some(Pass::Sv)]
                        } else {
                            vec![None]
                        };
                        for p in passes {
                            let h = match spec.check().ok().and_then(|_| emit_sdpa(&spec, p).ok()) {
                                Some(b) => format!("{:016x} {}", fnv(&b), b.len()),
                                None => "declined".into(),
                            };
                            all =
                                fnv(&[all.to_le_bytes(), fnv(h.as_bytes()).to_le_bytes()].concat());
                            lines += 1;
                            println!("row d={d} sink={sink} {form:?} qk={qk_group} spill={spill} {p:?}: {h}");
                        }
                    }
                }
            }
            for rows in 1..=8usize {
                for qk_group in [0u8, 8, 16] {
                    for qk_pair in [1u8, 2] {
                        for dim_blocks in [0usize, 1, 2] {
                            for (full, window) in
                                [(false, 0usize), (true, 0), (false, 64), (true, 64)]
                            {
                                let spec = SdpaPrefillSpec {
                                    shape: sh,
                                    rows,
                                    row_stride: 16 * d,
                                    out_row_stride: 16 * d,
                                    exp: ExpStrategy::Inline,
                                    pf_dist: SDPA_PF_DIST_PREFILL,
                                    dim_blocks,
                                    qk_group,
                                    qk_pair,
                                    kv_capacity: 4096,
                                    full,
                                    window,
                                    sv_tile: 0,
                                };
                                let h = match spec
                                    .check()
                                    .ok()
                                    .and_then(|_| emit_sdpa_prefill(&spec).ok())
                                {
                                    Some(b) => format!("{:016x} {}", fnv(&b), b.len()),
                                    None => "declined".into(),
                                };
                                all =
                                    fnv(&[all.to_le_bytes(), fnv(h.as_bytes()).to_le_bytes()]
                                        .concat());
                                lines += 1;
                                println!(
                                    "block d={d} sink={sink} rows={rows} qk={qk_group} pair={qk_pair} nb={dim_blocks} full={full} window={window}: {h}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!("DIGEST {lines} specs: {all:016x}");
}

/// The vision tower's patches: 7 176 image tokens, four patches each.
const N: usize = 28_704;
const HEADS: usize = 16;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Arm {
    d: usize,
    rows: usize,
    nb: usize,
    qk: u8,
    pair: u8,
    tile: usize,
}

/// Seconds for `blocks` blocks of `rows` queries per head (from row 0), over the `N` keys of each
/// head, on `threads` threads; `shared`: every thread on the same head (blocks dealt cyclically),
/// else thread `t` on head `t % HEADS`.
#[allow(clippy::too_many_arguments)]
fn run(
    f: SdpaPrefillFn,
    arm: &Arm,
    scratch: usize,
    q: &[u16],
    kv: &[u16],
    out: &mut [u16],
    blocks: usize,
    threads: usize,
    shared: bool,
) -> f64 {
    let d = arm.d;
    let qd = HEADS * d;
    let qp = q.as_ptr() as usize;
    let kp = kv.as_ptr() as usize;
    let op = out.as_mut_ptr() as usize;
    let rows = arm.rows;
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                let mut scores = vec![0f32; scratch + 64];
                let (head, first, step) = if shared {
                    (0, t, threads)
                } else {
                    (t % HEADS, 0, 1)
                };
                let k = kp + head * 2 * N * d * 2;
                let v = k + N * d * 2;
                let mut b = first;
                while b < blocks {
                    let r0 = b * rows;
                    unsafe {
                        f(
                            (qp + (r0 * qd + head * d) * 2) as *const u16,
                            k as *const u16,
                            v as *const u16,
                            scores.as_mut_ptr(),
                            (op + (r0 * qd + head * d) * 2) as *mut u16,
                            N as u64,
                            1.0 / (d as f32).sqrt(),
                            0.0,
                        );
                    }
                    b += step;
                }
            });
        }
    });
    t0.elapsed().as_secs_f64()
}

/// The block kernel without mask on the tower's shape, arms alternated in one process, best of
/// `REPS`.
///
/// `ARMS` (default all): `d:rows:nb:qk:pair[:tile]` separated by commas. `BLOCKS` per head (default
/// 96), `REPS` (default 3), `THREADS` (default 8).
#[test]
#[ignore = "a measurement, run by hand"]
fn bench_sdpa_block_full() {
    if !crate::jit::gemm::is_supported() {
        eprintln!("no AVX-512 BF16: skipped");
        return;
    }
    let arms: Vec<Arm> = std::env::var("ARMS")
        .unwrap_or_else(|_| "64:8:0:16:1,72:8:0:16:1,64:8:0:8:1,72:8:0:8:1".into())
        .split(',')
        .map(|a| {
            let p: Vec<usize> = a.split(':').map(|x| x.parse().unwrap()).collect();
            Arm {
                d: p[0],
                rows: p[1],
                nb: p[2],
                qk: p[3] as u8,
                pair: p[4] as u8,
                tile: p.get(5).copied().unwrap_or(0),
            }
        })
        .collect();
    let blocks = env_usize("BLOCKS", 96);
    let reps = env_usize("REPS", 3);
    let threads = env_usize("THREADS", 8);
    let dmax = arms.iter().map(|a| a.d).max().unwrap();
    let rmax = arms.iter().map(|a| a.rows).max().unwrap();
    let mut seed = 3u64;
    let mut rnd = |amp: f32| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((seed >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * amp;
        (x.to_bits() >> 16) as u16
    };
    let qrows = blocks * rmax;
    let q: Vec<u16> = (0..qrows * HEADS * dmax).map(|_| rnd(1.0)).collect();
    let kv: Vec<u16> = (0..HEADS * 2 * N * dmax).map(|_| rnd(1.0)).collect();
    let mut out = vec![0u16; qrows * HEADS * dmax + 64];
    let pages: Vec<_> = arms
        .iter()
        .map(|a| {
            let spec = SdpaPrefillSpec {
                shape: shape(a.d),
                rows: a.rows,
                row_stride: HEADS * a.d,
                out_row_stride: HEADS * a.d,
                exp: crate::jit::sdpa::sdpa_exp(),
                pf_dist: SDPA_PF_DIST_PREFILL,
                dim_blocks: a.nb,
                qk_group: a.qk,
                qk_pair: a.pair,
                kv_capacity: N,
                full: true,
                window: 0,
                sv_tile: a.tile,
            };
            spec.check().unwrap_or_else(|e| {
                panic!(
                    "d={} rows={} nb={} qk={} pair={}: {e:?}",
                    a.d, a.rows, a.nb, a.qk, a.pair
                )
            });
            let bytes = emit_sdpa_prefill(&spec).unwrap();
            let page =
                crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(&bytes)).unwrap();
            (
                page,
                spec.scratch_len(),
                format!("{:?} tile={:?}", spec.sv_groups(), spec.sv_tile_len()),
                bytes.len(),
            )
        })
        .collect();
    println!("N={N} heads={HEADS} blocks/head={blocks} reps={reps} threads={threads}");
    for (mode, th, shared) in [
        ("1 thread", 1, false),
        ("threads, one head each", threads, false),
        ("threads, one head shared", threads, true),
    ] {
        let mut best = vec![f64::INFINITY; arms.len()];
        for _ in 0..reps {
            for (i, a) in arms.iter().enumerate() {
                let f: SdpaPrefillFn = unsafe { std::mem::transmute(pages[i].0.as_ptr()) };
                let t = run(f, a, pages[i].1, &q, &kv, &mut out, blocks, th, shared);
                best[i] = best[i].min(t);
            }
        }
        for (i, a) in arms.iter().enumerate() {
            // Calls made: `blocks` per thread (own head) or `blocks` in all (shared head).
            let calls = if shared { blocks } else { blocks * th };
            let per_call = best[i]
                / (if shared {
                    calls as f64 / th as f64
                } else {
                    blocks as f64
                });
            let flops = 4.0 * (N * a.d * a.rows) as f64 * calls as f64;
            println!(
                "{mode:<26} d={} rows={} nb={} qk={} pair={} code={}B: {:8.3} ms/call/thread, {:7.1} GFLOP/s, {:6.2} ns/(row·key)",
                a.d,
                a.rows,
                pages[i].2,
                a.qk,
                a.pair,
                pages[i].3,
                per_call * 1e3,
                flops / best[i] / 1e9,
                per_call * 1e9 / (a.rows * N) as f64,
            );
        }
    }
}
