//! The block attention with a sliding window (`SdpaPrefillSpec::window`, LFM3's sliding layers)
//! against the row attention with the same window (`AttentionSpec::window`, Gemma 4's bounds),
//! through the generated traversals of the build's target, on a cache written by the generated KV
//! append (with its R16 copies on AArch64).
//!
//! The prompt runs in passes cut before, on and after the window (the first row whose window drops
//! a key, `window`, and the rows around it), and in one pass; passes whose length is not a multiple
//! of the block's rows end on a partial tile (the row kernel's), and every pass after the first has
//! `past > 0`.
//!
//! - x86-64, with AVX-512 (the `zmm` block) or without (the AVX2 block, `emit_sdpa_avx2.rs`): the
//!   block kernel has the bits of the decode kernel; every row, bit for bit, against the row path.
//!   A block kernel that declines the window there is a failure, not a skip.
//! - AArch64: the block kernel's bits are those of its two GEMMs, not of the decode kernel
//!   (`sdpa_block.rs`). Bit for bit: the windowed block against the causal block on the rows whose
//!   window holds every key, and against itself whatever the cuts; past the window, against the row
//!   path to a bound (and against the causal block, which must differ there: the window is
//!   applied).
//! - No block kernel on this ISA (another target): said, and the row path alone has nothing to
//!   compare.

// `assert!(!cfg!(...))`: the targets allowed to decline a kernel, checked where it declines.
#![allow(clippy::assertions_on_constants)]
use crate::jit::arch::{KernelSet, Native};
use crate::jit::execution::abi::{AttentionCall, AttentionSpec, KvCall, KvSpec};
use crate::jit::ir::SdpaShape;

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}
fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len().max(16), |m| {
        m[..bytes.len()].copy_from_slice(bytes)
    })
    .unwrap()
}
type Selector = crate::jit::arch::generated_fn!(*const AttentionCall, usize, *mut f32);
type Append = crate::jit::arch::generated_fn!(*const KvCall, usize);

/// The attention's geometry: query heads, query heads per KV head, prompt rows, positions reserved
/// per KV head, head width, scale of the scores.
#[derive(Clone, Copy)]
struct G {
    heads: usize,
    gqa: usize,
    t: usize,
    cap: usize,
    hd: usize,
    scale: f32,
}
/// The test's: four heads on two KV heads, 150 rows, heads of 128.
const SMALL: G = G {
    heads: 4,
    gqa: 2,
    t: 150,
    cap: 256,
    hd: 128,
    scale: 0.088_388_35,
};

struct Cache {
    keys: Vec<u16>,
    values: Vec<u16>,
    keys_r16: Vec<u16>,
    values_r16: Vec<u16>,
}

/// The whole prompt's keys and values into the cache, by the generated append.
#[allow(non_snake_case)]
fn cache(g: G, seed: &mut u64) -> Cache {
    let G {
        t: T,
        cap: CAP,
        hd: HD,
        ..
    } = g;
    let KV = g.heads / g.gqa;
    let r16 = cfg!(target_arch = "aarch64");
    let stride = CAP * HD;
    let spec = KvSpec {
        heads: KV,
        head_dim: HD,
        stride,
        capacity: CAP,
        max_batch: T,
        r16,
    };
    let kernel = Native::kv_append(spec).unwrap();
    let page = exec(&kernel.bytes);
    let run: Append = unsafe { std::mem::transmute(page.as_ptr()) };
    let input_k: Vec<u16> = (0..T * KV * HD).map(|_| bf16(rng(seed) * 2.0)).collect();
    let input_v: Vec<u16> = (0..T * KV * HD).map(|_| bf16(rng(seed))).collect();
    let mut c = Cache {
        keys: vec![0; KV * stride],
        values: vec![0; KV * stride],
        keys_r16: vec![0; if r16 { KV * stride } else { 0 }],
        values_r16: vec![0; if r16 { KV * stride } else { 0 }],
    };
    let (rows, past) = (T, 0usize);
    let call = KvCall {
        keys: c.keys.as_mut_ptr(),
        values: c.values.as_mut_ptr(),
        input_keys: input_k.as_ptr(),
        input_values: input_v.as_ptr(),
        rows: &rows,
        past: &past,
        keys_r16: if r16 {
            c.keys_r16.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
        values_r16: if r16 {
            c.values_r16.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
    };
    for h in 0..KV {
        unsafe { run(&call, h) };
    }
    c
}

/// The attention of rows `0..T` in the passes `cuts` (their first rows, then `T`), by the traversal
/// of `spec` with `prefill` (or the row kernel alone).
#[allow(non_snake_case)]
fn attend(
    g: G,
    c: &Cache,
    q: &[u16],
    cuts: &[usize],
    window: usize,
    prefill: Option<&crate::jit::sdpa::CompiledSdpaPrefill>,
) -> Vec<u16> {
    let G {
        heads: HEADS,
        gqa: GQA,
        t: T,
        cap: CAP,
        hd: HD,
        scale,
    } = g;
    let sdpa = crate::jit::sdpa::head_kernel(SdpaShape {
        head_dim: HD,
        sink: false,
    })
    .unwrap()
    .unwrap();
    let spec = AttentionSpec {
        heads: HEADS,
        gqa: GQA,
        head_dim: HD,
        q_row: HEADS * HD,
        q_head: HD,
        out_row: HEADS * HD,
        grouped: false,
        kv_task: false,
        window,
        whole_block: false,
        ahead: 0,
        segments: false,
    };
    let kernel = Native::attention_tile_selector(&sdpa, prefill, spec).unwrap();
    let page = exec(&kernel.bytes);
    let run: Selector = unsafe { std::mem::transmute(page.as_ptr()) };
    let chunk = prefill.map_or(1, |p| p.rows());
    let mut scores = vec![0f32; prefill.map_or(CAP.next_multiple_of(16), |p| p.scratch_len()) + 64];
    let mut out = vec![0u16; T * HEADS * HD];
    for w in cuts
        .iter()
        .copied()
        .chain([T])
        .collect::<Vec<_>>()
        .windows(2)
    {
        let (past, rows) = (w[0], w[1] - w[0]);
        let call = AttentionCall {
            q: q[past * HEADS * HD..].as_ptr(),
            keys: c.keys.as_ptr(),
            values: c.values.as_ptr(),
            output: out[past * HEADS * HD..].as_mut_ptr(),
            rows: &rows,
            past: &past,
            kv_stride: CAP * HD,
            scale,
            sink: 0.0,
            keys_r16: if c.keys_r16.is_empty() {
                std::ptr::null()
            } else {
                c.keys_r16.as_ptr()
            },
            values_r16: if c.values_r16.is_empty() {
                std::ptr::null()
            } else {
                c.values_r16.as_ptr()
            },
            segments: std::ptr::null(),
        };
        let tasks = rows.div_ceil(chunk) * HEADS;
        for t in 0..tasks {
            unsafe { run(&call, t, scores.as_mut_ptr()) };
        }
    }
    out
}

#[test]
#[allow(non_snake_case)]
fn windowed_block_attention_matches_rows() {
    let G {
        heads: HEADS,
        t: T,
        cap: CAP,
        hd: HD,
        ..
    } = SMALL;
    let shape = SdpaShape {
        head_dim: HD,
        sink: false,
    };
    let rows = crate::jit::sdpa::sdpa_rows();
    let Some(causal) =
        crate::jit::sdpa::prefill_kernel(shape, rows, HEADS * HD, HEADS * HD, CAP).unwrap()
    else {
        assert!(
            !cfg!(target_arch = "x86_64"),
            "windowed block attention: the causal block kernel declines on x86-64"
        );
        eprintln!("windowed block attention: no block kernel on this ISA (the row kernel serves every layer)");
        return;
    };
    let mut seed = 0x5eed_u64;
    let c = cache(SMALL, &mut seed);
    let q: Vec<u16> = (0..T * HEADS * HD)
        .map(|_| bf16(rng(&mut seed) * 2.0))
        .collect();
    let mut checked = 0usize;
    for window in [40usize, 33, 64, 200] {
        let Some(blocked) =
            crate::jit::sdpa::get_or_compile_prefill(crate::jit::emit_sdpa::SdpaPrefillSpec {
                window,
                ..causal.spec()
            })
            .unwrap()
        else {
            assert!(
                !cfg!(target_arch = "x86_64"),
                "windowed block attention: the block kernel declines window {window} on x86-64"
            );
            eprintln!("windowed block attention: the block kernel of this ISA takes no window (the row kernel serves sliding layers)");
            return;
        };
        // Cuts before, on and after the window: a pass from 0 to window − 7, one ending on window,
        // one from window + 1, a long one, a short tail; then the whole prompt in one pass.
        // A window longer than the prompt (200): passes of 5, 72, 72 and 1 rows, of 64 and
        // 86.
        let cut_sets: [Vec<usize>; 3] = if window < T {
            [
                vec![
                    0,
                    window - 7,
                    window - 1,
                    window,
                    window + 1,
                    2 * window + 3,
                    T - 5,
                ],
                vec![0, 5, window - 2, window + 9, T - 1],
                vec![0],
            ]
        } else {
            [vec![0, 5, 77, T - 1], vec![0, 64], vec![0]]
        };
        let rowwise = attend(SMALL, &c, &q, &[0], window, None);
        let mut first: Option<Vec<u16>> = None;
        for cuts in &cut_sets {
            let got = attend(SMALL, &c, &q, cuts, window, Some(&blocked));
            if cfg!(target_arch = "x86_64") {
                for r in 0..T {
                    let (a, b) = (
                        &got[r * HEADS * HD..][..HEADS * HD],
                        &rowwise[r * HEADS * HD..][..HEADS * HD],
                    );
                    assert_eq!(
                        a, b,
                        "window {window} cuts {cuts:?}: row {r} differs from the row path"
                    );
                }
            } else {
                // The same bits whatever the cuts.
                if let Some(f0) = &first {
                    assert_eq!(
                        &got, f0,
                        "window {window} cuts {cuts:?}: the bits depend on the cuts"
                    );
                } else {
                    first = Some(got.clone());
                }
                let causal_out = attend(SMALL, &c, &q, cuts, 0, Some(&causal));
                let (mut worst, mut differ) = (0f32, 0usize);
                for r in 0..T {
                    let (a, b, z) = (
                        &got[r * HEADS * HD..][..HEADS * HD],
                        &rowwise[r * HEADS * HD..][..HEADS * HD],
                        &causal_out[r * HEADS * HD..][..HEADS * HD],
                    );
                    if r < window {
                        assert_eq!(a, z, "window {window} cuts {cuts:?}: row {r} (whole window) differs from the causal block");
                    } else {
                        differ += usize::from(a != z);
                    }
                    for (x, y) in a.iter().zip(b) {
                        worst = worst.max((f(*x) - f(*y)).abs());
                    }
                }
                assert_eq!(
                    differ,
                    T.saturating_sub(window),
                    "window {window} cuts {cuts:?}: rows past the window equal to the causal block"
                );
                assert!(
                    worst <= 1.0 / 64.0,
                    "window {window} cuts {cuts:?}: {worst} from the row path"
                );
                eprintln!("window {window} cuts {cuts:?}: largest gap to the row path {worst}");
            }
            checked += T;
        }
    }
    eprintln!(
        "windowed block attention: {checked} rows checked ({})",
        if cfg!(target_arch = "x86_64") {
            "bit for bit against the row path"
        } else {
            "NEON"
        }
    );
}

/// Gemma 4's two layer kinds through the block kernel (`lower_gemma_layer`), against the row path:
/// a full layer (heads of 512, causal, eight query heads on one KV head as on E2B) and a sliding
/// layer (heads of 256, window 512, eight query heads on two KV heads as on E4B, 700 rows), the
/// scale at one (queries and keys normed per head). Passes cut before, on and after the window, and
/// in one pass. x86-64: every row bit for bit against the row path (AVX-512 and AVX2, both layers;
/// a block kernel that declines the window is a failure there). AArch64: the same bits whatever the
/// cuts, the sliding layer bit for bit against the causal block on the rows whose window holds
/// every key, and both against the row path to a bound.
#[test]
#[allow(non_snake_case)]
fn gemma_block_attention_matches_rows() {
    let rows = crate::jit::sdpa::sdpa_rows();
    let full = G {
        heads: 8,
        gqa: 8,
        t: 300,
        cap: 320,
        hd: 512,
        scale: 1.0,
    };
    let sliding = G {
        heads: 8,
        gqa: 4,
        t: 700,
        cap: 768,
        hd: 256,
        scale: 1.0,
    };
    let window = 512usize;
    for (name, g, w, cut_sets) in [
        (
            "full, head 512",
            full,
            0usize,
            vec![
                vec![0usize, 5, 100, 233, 299],
                vec![0, 1, 2, 3, 150],
                vec![0],
            ],
        ),
        (
            "sliding, head 256, window 512",
            sliding,
            window,
            vec![
                vec![0, window - 7, window - 1, window, window + 1, 640, 695],
                vec![0, 300, window - 2, window + 9, 699],
                vec![0],
            ],
        ),
    ] {
        let G {
            heads: HEADS,
            t: T,
            cap: CAP,
            hd: HD,
            ..
        } = g;
        let row = HEADS * HD;
        let shape = SdpaShape {
            head_dim: HD,
            sink: false,
        };
        let causal = crate::jit::sdpa::prefill_kernel(shape, rows, row, row, CAP).unwrap();
        let Some(causal) = causal else {
            assert!(
                !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
                "{name}: the block kernel declines"
            );
            eprintln!("{name}: no block kernel on this ISA");
            continue;
        };
        let blocked = if w > 0 {
            match crate::jit::sdpa::get_or_compile_prefill(crate::jit::emit_sdpa::SdpaPrefillSpec {
                window: w,
                ..causal.spec()
            })
            .unwrap()
            {
                Some(p) => p,
                None => {
                    assert!(
                        !cfg!(target_arch = "x86_64"),
                        "{name}: the block kernel declines the window on x86-64"
                    );
                    eprintln!("{name}: the block kernel of this ISA takes no window (the row kernel serves sliding layers)");
                    continue;
                }
            }
        } else {
            causal.clone()
        };
        let mut seed = 0x6e33_u64 + HD as u64;
        let c = cache(g, &mut seed);
        // Queries a quarter of the keys' scale: at scale one, wider scores make the softmax so
        // sharp that the keys before the window weigh nothing on some rows (on NEON, three
        // rows of 188 came out equal to the causal block), and the test could no longer see
        // that the window is applied.
        let q: Vec<u16> = (0..T * row).map(|_| bf16(rng(&mut seed) * 0.25)).collect();
        let rowwise = attend(g, &c, &q, &[0], w, None);
        let mut first: Option<Vec<u16>> = None;
        for cuts in &cut_sets {
            let got = attend(g, &c, &q, cuts, w, Some(&blocked));
            if cfg!(target_arch = "x86_64") {
                for r in 0..T {
                    assert_eq!(
                        &got[r * row..][..row],
                        &rowwise[r * row..][..row],
                        "{name} cuts {cuts:?}: row {r} differs from the row path"
                    );
                }
                eprintln!("{name} cuts {cuts:?}: {T} rows bit for bit against the row path");
                continue;
            }
            if let Some(f0) = &first {
                assert_eq!(
                    &got, f0,
                    "{name} cuts {cuts:?}: the bits depend on the cuts"
                );
            } else {
                first = Some(got.clone());
            }
            if w > 0 {
                let causal_out = attend(g, &c, &q, cuts, 0, Some(&causal));
                let mut differ = 0usize;
                for r in 0..T {
                    let (a, z) = (&got[r * row..][..row], &causal_out[r * row..][..row]);
                    if r < w {
                        assert_eq!(a, z, "{name} cuts {cuts:?}: row {r} (whole window) differs from the causal block");
                    } else {
                        differ += usize::from(a != z);
                    }
                }
                assert_eq!(
                    differ,
                    T - w,
                    "{name} cuts {cuts:?}: rows past the window equal to the causal block"
                );
            }
            let worst = got
                .iter()
                .zip(&rowwise)
                .map(|(x, y)| (f(*x) - f(*y)).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 1.0 / 64.0,
                "{name} cuts {cuts:?}: {worst} from the row path"
            );
            eprintln!("{name} cuts {cuts:?}: largest gap to the row path {worst}");
        }
    }
}

/// The attention of one LFM3-1.7B sliding layer over its 1 943-token prompt (sixteen heads on four
/// KV heads, window 256), one participant, in passes of 512 rows: the row kernel against the
/// windowed block kernel, and the causal block kernel for scale, three rounds interleaved. `cargo
/// test --release --lib window_attn_speed -- --ignored --nocapture`.
#[test]
#[ignore = "a speed measurement, run by hand"]
#[allow(non_snake_case)]
fn window_attn_speed() {
    let g = G {
        heads: 16,
        gqa: 4,
        t: 1943,
        cap: 2048,
        hd: 128,
        scale: 0.088_388_35,
    };
    let HD = g.hd;
    let (window, pass) = (256usize, 512usize);
    let shape = SdpaShape {
        head_dim: HD,
        sink: false,
    };
    let rows = crate::jit::sdpa::sdpa_rows();
    let Some(causal) =
        crate::jit::sdpa::prefill_kernel(shape, rows, g.heads * HD, g.heads * HD, g.cap).unwrap()
    else {
        assert!(
            !cfg!(target_arch = "x86_64"),
            "window_attn_speed: the causal block kernel declines on x86-64"
        );
        eprintln!("window_attn_speed: no block kernel on this ISA");
        return;
    };
    let Some(blocked) =
        crate::jit::sdpa::get_or_compile_prefill(crate::jit::emit_sdpa::SdpaPrefillSpec {
            window,
            ..causal.spec()
        })
        .unwrap()
    else {
        assert!(
            !cfg!(target_arch = "x86_64"),
            "window_attn_speed: the block kernel declines the window on x86-64"
        );
        eprintln!("window_attn_speed: the block kernel of this ISA takes no window");
        return;
    };
    let mut seed = 0x5eed_u64;
    let c = cache(g, &mut seed);
    let q: Vec<u16> = (0..g.t * g.heads * HD)
        .map(|_| bf16(rng(&mut seed) * 2.0))
        .collect();
    let cuts: Vec<usize> = (0..g.t).step_by(pass).collect();
    for round in 1..=3 {
        for (name, w, p) in [
            ("rows, window", window, None),
            ("block, window", window, Some(&*blocked)),
            ("block, causal", 0, Some(&*causal)),
        ] {
            let t0 = std::time::Instant::now();
            let out = attend(g, &c, &q, &cuts, w, p);
            let dt = t0.elapsed();
            std::hint::black_box(out);
            eprintln!("round {round} {name}: {:.1} ms", dt.as_secs_f64() * 1e3);
        }
    }
}

/// The gap of two bf16 in steps of bf16 at the larger of the two (`|a − b| / ulp(max(|a|, |b|))`,
/// the ulp of a bf16 in `[2^e, 2^(e+1))` being `2^(e−7)`).
fn bf16_steps(a: u16, b: u16) -> f32 {
    let m = f(a).abs().max(f(b).abs());
    if m == 0.0 {
        return 0.0;
    }
    (f(a) - f(b)).abs() / 2f32.powi(m.log2().floor() as i32 - 7)
}

/// The unmasked attention of rows `0..T` in one pass (the encoders', the vision tower's:
/// `whole_block`), by the traversal of the build's target with `prefill` (or the row kernel alone):
/// every key, or each row's input only (`segments`, `[first, end)` per row), or `|q − k| < window`
/// (`ahead = window − 1`).
#[allow(non_snake_case)]
fn attend_full(
    g: G,
    c: &Cache,
    q: &[u16],
    segments: Option<&[u32]>,
    window: usize,
    prefill: Option<&crate::jit::sdpa::CompiledSdpaPrefill>,
) -> Vec<u16> {
    let G {
        heads: HEADS,
        gqa: GQA,
        t: T,
        cap: CAP,
        hd: HD,
        scale,
    } = g;
    let sdpa = crate::jit::sdpa::head_kernel(SdpaShape {
        head_dim: HD,
        sink: false,
    })
    .unwrap()
    .unwrap();
    let spec = AttentionSpec {
        heads: HEADS,
        gqa: GQA,
        head_dim: HD,
        q_row: HEADS * HD,
        q_head: HD,
        out_row: HEADS * HD,
        grouped: false,
        kv_task: false,
        window,
        whole_block: true,
        ahead: window.saturating_sub(1),
        segments: segments.is_some(),
    };
    let kernel = Native::attention_tile_selector(&sdpa, prefill, spec).unwrap();
    let page = exec(&kernel.bytes);
    let run: Selector = unsafe { std::mem::transmute(page.as_ptr()) };
    let chunk = prefill.map_or(1, |p| p.rows());
    // The scratch the kernels claim, exactly, then a sentinel: a kernel writing past its
    // `scratch_len` fails here.
    const SENTINEL: usize = 256;
    let claimed = prefill
        .map_or(0, |p| p.scratch_len())
        .max(CAP.next_multiple_of(16));
    let mut scores = vec![0f32; claimed + SENTINEL];
    scores[claimed..].fill(f32::from_bits(0x7fc0_beef));
    let mut out = vec![0u16; T * HEADS * HD];
    let (past, rows) = (0usize, T);
    let call = AttentionCall {
        q: q.as_ptr(),
        keys: c.keys.as_ptr(),
        values: c.values.as_ptr(),
        output: out.as_mut_ptr(),
        rows: &rows,
        past: &past,
        kv_stride: CAP * HD,
        scale,
        sink: 0.0,
        keys_r16: if c.keys_r16.is_empty() {
            std::ptr::null()
        } else {
            c.keys_r16.as_ptr()
        },
        values_r16: if c.values_r16.is_empty() {
            std::ptr::null()
        } else {
            c.values_r16.as_ptr()
        },
        segments: segments.map_or(std::ptr::null(), <[u32]>::as_ptr),
    };
    let tasks = rows.div_ceil(chunk) * HEADS;
    for t in 0..tasks {
        unsafe { run(&call, t, scores.as_mut_ptr()) };
    }
    assert!(
        scores[claimed..].iter().all(|x| x.to_bits() == 0x7fc0_beef),
        "the attention wrote past its scratch ({claimed} f32)"
    );
    out
}

/// The block kernel without the causal mask (`SdpaPrefillSpec::full`) against the row path on the
/// same traversal: every key (the vision tower's patches, one input of an encoder), several inputs
/// end to end (`segments`, tiles across two inputs taking the row kernel), and the bidirectional
/// window of EmbeddingGemma 2's sliding layers (`|q − k| < window`: odd and even windows, one just
/// under the sequence, one past it). Heads of 128 (four on two KV heads), of 64 (the vision tower's
/// size) and of 256, 150 rows (the last tile partial: the row kernel). On x86-64, with AVX-512 (the
/// zmm block) or AVX2 alone (`emit_sdpa_avx2.rs`), the block has the bits of the row path and must
/// take every case (a refusal fails); on AArch64 its
/// bits are not the row path's (two GEMMs, `P` in bf16): the largest gap, in steps of bf16, is
/// printed and bounded.
#[test]
#[allow(non_snake_case)]
fn full_block_attention_matches_rows() {
    #[cfg(target_arch = "x86_64")]
    let x86_block = crate::jit::gemm::is_supported() || crate::jit::emit_sdpa_avx2::is_supported();
    #[cfg(not(target_arch = "x86_64"))]
    let x86_block = false;
    let isa = if crate::jit::gemm::is_supported() {
        "AVX-512"
    } else {
        "AVX2"
    };
    let mut checked = 0usize;
    for g in [
        SMALL,
        G {
            heads: 4,
            gqa: 1,
            t: 150,
            cap: 160,
            hd: 64,
            scale: 0.125,
        },
        G {
            heads: 2,
            gqa: 1,
            t: 150,
            cap: 160,
            hd: 256,
            scale: 0.0625,
        },
    ] {
        let G {
            heads: HEADS,
            t: T,
            cap: CAP,
            hd: HD,
            ..
        } = g;
        let shape = SdpaShape {
            head_dim: HD,
            sink: false,
        };
        let qk = crate::jit::sdpa::head_kernel(shape)
            .unwrap()
            .unwrap()
            .spec()
            .qk_group;
        let rows = crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS;
        let Some(full) =
            crate::jit::sdpa::prefill_kernel_full(shape, rows, HEADS * HD, HEADS * HD, CAP, qk, 0)
                .unwrap()
        else {
            assert!(
                !x86_block,
                "head_dim {HD}: the {isa} unmasked block refused"
            );
            eprintln!("full block attention: no unmasked block kernel on this ISA (the row kernel serves)");
            return;
        };
        let mut seed = 0xf011_u64 + HD as u64;
        let c = cache(g, &mut seed);
        let q: Vec<u16> = (0..T * HEADS * HD)
            .map(|_| bf16(rng(&mut seed) * 2.0))
            .collect();
        // Three inputs end to end, none on a multiple of sixteen.
        let mut segments = Vec::new();
        for (first, end) in [(0u32, 37u32), (37, 107), (107, T as u32)] {
            for _ in first..end {
                segments.extend([first, end]);
            }
        }
        type Case<'a> = (
            String,
            Option<&'a [u32]>,
            usize,
            std::sync::Arc<crate::jit::sdpa::CompiledSdpaPrefill>,
        );
        let mut cases: Vec<Case> = vec![
            ("every key".into(), None, 0, full.clone()),
            ("three inputs".into(), Some(&segments[..]), 0, full.clone()),
        ];
        // 33 and 145 odd, 145 just under the sequence (only the middle rows see every key), 200
        // past it.
        for window in [33usize, 64, 145, 200] {
            if let Some(w) = crate::jit::sdpa::prefill_kernel_full(
                shape,
                rows,
                HEADS * HD,
                HEADS * HD,
                CAP,
                qk,
                window,
            )
            .unwrap()
            {
                cases.push((format!("|q - k| < {window}"), None, window, w));
            } else {
                assert!(
                    !x86_block,
                    "head_dim {HD}, window {window}: the {isa} unmasked block refused the window"
                );
                eprintln!("full block attention: no window on this ISA's unmasked block (the row kernel serves sliding layers)");
            }
        }
        let every_key = attend_full(g, &c, &q, None, 0, None);
        for (name, seg, window, kernel) in &cases {
            let rowwise = attend_full(g, &c, &q, *seg, *window, None);
            let got = attend_full(g, &c, &q, *seg, *window, Some(kernel));
            eprintln!(
                "head_dim {HD}, {name}: block of {} rows, window {}, full {}",
                kernel.rows(),
                kernel.spec().window,
                kernel.spec().full
            );
            // A window shorter than the sequence hides keys: its rows differ from every key's.
            if (1..T).contains(window) {
                let row = HEADS * HD;
                let differ = (0..T)
                    .filter(|&r| got[r * row..][..row] != every_key[r * row..][..row])
                    .count();
                assert!(
                    differ > 0,
                    "head_dim {HD}, {name}: the window changes nothing"
                );
                eprintln!("head_dim {HD}, {name}: {differ} of {T} rows differ from every key's");
            }
            // Steps on the outputs of at least 1/16 (below, a step shrinks with the cancellation of
            // the sum).
            let (mut steps, mut worst, mut equal) = (0f32, 0f32, 0usize);
            for (x, y) in got.iter().zip(&rowwise) {
                if f(*y).abs() >= 0.0625 {
                    steps = steps.max(bf16_steps(*x, *y));
                }
                worst = worst.max((f(*x) - f(*y)).abs());
                equal += usize::from(x == y);
            }
            eprintln!(
                "head_dim {HD}, {name}: largest gap to the row path {worst} ({} bf16 steps of [0.5, 1)), {steps} bf16 steps on the outputs of at least 1/16, {:.1} % of the outputs equal",
                worst * 256.0,
                100.0 * equal as f64 / got.len() as f64
            );
            if x86_block {
                assert!(
                    got == rowwise,
                    "head_dim {HD}, {name}: the {isa} block is not the row path bit for bit"
                );
            }
            assert!(
                worst <= 1.0 / 64.0,
                "head_dim {HD}, {name}: {worst} from the row path"
            );
            checked += T;
        }
    }
    eprintln!("full block attention: {checked} rows checked");
}
