//! What the engine and the server both compute on the media of a request, in one place: the
//! resampling of a recording and the size an image is resized to.

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The coefficients of a resampling table at most (16 MiB of f32).
const MAX_KERNEL: usize = 1 << 22;

/// Torchaudio's `resample` with
/// `sinc_interp_kaiser` at resampy's `kaiser_best` (`lowpass_filter_width` 64, `rolloff`
/// 0.9475937167399596, `beta` 14.769656459379492), sample for sample: the reduced ratio `from /
/// to`, per output phase a Kaiser-windowed sinc of `2·width + from` taps over the input padded by
/// `width` zeros before, the output cut to `⌈to · len / from⌉`. Of the resamplers tried for
/// EmbeddingGemma 2's recordings at 44.1 and 48 kHz, the closest to librosa's `soxr_hq` (what
/// sentence-transformers reads them with): 1 − cos 2.2e-3 on average against 1.7e-2 for
/// the engine's `resample`. The coefficients in f64, the sums in f32, the output split over the
/// host's threads.
pub fn resample_kaiser_best(x: &[f32], from: usize, to: usize) -> Result<Vec<f32>, String> {
    if from == to {
        return Ok(x.to_vec());
    }
    let g = gcd(from, to);
    let (orig, new) = (from / g, to / g);
    let (lw, rolloff, beta) = (64.0f64, 0.9475937167399596f64, 14.769656459379492f64);
    let base = orig.min(new) as f64 * rolloff;
    let width = (lw * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    // The table holds `new · taps` coefficients: a rate whose ratio to `to` does not reduce (44 101
    // Hz to 16 kHz: 16 000 phases of 44 473 taps, 2.8 GB) is refused instead of computed. The usual
    // rates (8 to 192 kHz) need fewer than 400 000.
    if new.saturating_mul(taps) > MAX_KERNEL {
        return Err(format!(
            "a rate of {from} Hz is not resampled to {to} Hz here: convert the file first (ffmpeg -i in -ar {to} -ac 1 out.wav)"
        ));
    }
    // I₀ by its series, to the precision of f64.
    let i0 = |x: f64| {
        let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
        while term > sum * 1e-17 {
            term *= (x / (2.0 * k)) * (x / (2.0 * k));
            sum += term;
            k += 1.0;
        }
        sum
    };
    let i0b = i0(beta);
    let scale = base / orig as f64;
    let kernels: Vec<f32> = (0..new)
        .flat_map(|p| {
            (0..taps).map(move |j| {
                let idx = (j as f64 - width as f64) / orig as f64;
                let t = ((-(p as f64)) / new as f64 + idx) * base;
                let t = t.clamp(-lw, lw);
                let r = t / lw;
                let w = i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
                let tp = t * std::f64::consts::PI;
                let s = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
                (s * w * scale) as f32
            })
        })
        .collect();
    let n_out = (new * x.len()).div_ceil(orig);
    let mut out = vec![0f32; n_out];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 16);
    let chunk = n_out.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (c, part) in out.chunks_mut(chunk).enumerate() {
            let kernels = &kernels;
            sc.spawn(move || {
                for (o, y) in part.iter_mut().enumerate() {
                    let j = c * chunk + o;
                    let (b, p) = (j / new, j % new);
                    let k = &kernels[p * taps..(p + 1) * taps];
                    // Input sample of tap 0: `b·orig − width`; the taps over the padding's zeros
                    // are skipped.
                    let start = (b * orig) as isize - width as isize;
                    let t0 = (-start).max(0) as usize;
                    let t1 = ((x.len() as isize - start).max(0) as usize).min(taps);
                    *y = if t0 < t1 {
                        k[t0..t1]
                            .iter()
                            .zip(&x[(start + t0 as isize) as usize..])
                            .map(|(w, v)| w * v)
                            .sum()
                    } else {
                        0.0
                    };
                }
            });
        }
    });
    Ok(out)
}

/// The size the Gemma 4 processor resizes an `h × w` image to (`get_aspect_ratio_preserving_size`):
/// both sides a multiple of `patch · pool`, at most `max_patches` patches.
pub fn target_size(
    h: usize,
    w: usize,
    patch: usize,
    max_patches: usize,
    pool: usize,
) -> Result<(usize, usize), String> {
    if h == 0 || w == 0 {
        return Err(format!("an image of {w} × {h} pixels"));
    }
    let total = (h * w) as f64;
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / total).sqrt();
    let side = pool * patch;
    let mut th = ((factor * h as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * w as f64) / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        return Err("image too small: it resizes to 0 × 0".into());
    }
    let max_side = (max_patches / (pool * pool)) * side;
    if th == 0 {
        th = side;
        tw = ((w / h) * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((h / w) * side).min(max_side);
    }
    Ok((th, tw))
}
