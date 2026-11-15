//! The images of a request: the `image_url` parts of the messages (OpenAI's shape, a `data:` URL;
//! the server fetches no address), decoded to RGB, and the soft tokens each takes as Gemma 4's
//! processor counts them.

use crate::backend::Vision;
use serde_json::Value;

/// The bytes of an image as sent (base64 decoded) at most.
pub const MAX_BYTES: usize = 32 << 20;

/// The pixels of an image at most (8192 × 8192).
pub const MAX_PIXELS: usize = 1 << 26;

/// One image of a request, at its own size.
pub struct Image {
    /// `h` rows of `w` pixels, 3 bytes each.
    pub rgb: Vec<u8>,
    pub h: usize,
    pub w: usize,
    /// FNV-1a over the size and the pixels: what tells two images apart in the tokens the server
    /// keeps (`worker.rs`).
    pub digest: u64,
    /// Its soft tokens when the engine counts them itself
    /// ([`crate::backend::Backend::image_soft_tokens`]), set by the worker; `None`: Gemma 4's
    /// rule.
    pub soft: Option<usize>,
}

/// The images of `messages`, in the order the templates write their placeholders: message by
/// message, the parts of each in order (a part `{"type": "image_url", "image_url": {"url": …}}`,
/// `"image_url": "…"` or `{"type": "image", …}` with one of those).
pub fn collect(messages: &Value) -> Result<Vec<Image>, String> {
    let mut out = Vec::new();
    for m in messages.as_array().into_iter().flatten() {
        for part in m["content"].as_array().into_iter().flatten() {
            if !matches!(part["type"].as_str(), Some("image_url" | "image")) {
                continue;
            }
            let url = part["image_url"]["url"]
                .as_str()
                .or_else(|| part["image_url"].as_str())
                .or_else(|| part["url"].as_str())
                .or_else(|| part["image"].as_str());
            let url = url.ok_or_else(|| format!("image {}: no URL in the part", out.len()))?;
            out.push(decode(url).map_err(|e| format!("image {}: {e}", out.len()))?);
        }
    }
    Ok(out)
}

/// A `data:image/…;base64,…` URL decoded to RGB.
pub fn decode(url: &str) -> Result<Image, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or("only data: URLs are accepted (the server fetches no address)")?;
    let (head, data) = rest.split_once(',').ok_or("a data: URL without a comma")?;
    if !head.ends_with(";base64") {
        return Err("a data: URL that is not base64".into());
    }
    if data.len() / 4 * 3 > MAX_BYTES {
        return Err(format!("more than {} MiB", MAX_BYTES >> 20));
    }
    let bytes = base64::decode(data.trim()).map_err(|e| format!("base64: {e}"))?;
    let reader = image::ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let (w, h) = reader.into_dimensions().map_err(|e| e.to_string())?;
    if (w as usize) * (h as usize) > MAX_PIXELS {
        return Err(format!("{w} × {h} pixels: at most {MAX_PIXELS}"));
    }
    let img = image::load_from_memory(&bytes)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    let (h, w) = (img.height() as usize, img.width() as usize);
    let rgb = img.into_raw();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for b in (h as u64)
        .to_le_bytes()
        .into_iter()
        .chain((w as u64).to_le_bytes())
        .chain(rgb.iter().copied())
    {
        digest = (digest ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(Image {
        rgb,
        h,
        w,
        digest,
        soft: None,
    })
}

/// The size the processor resizes an `h × w` image to (`get_aspect_ratio_preserving_size`): both
/// sides a multiple of `patch · pool`, at most `max_soft · pool²` patches.
pub fn target_size(h: usize, w: usize, v: Vision) -> Result<(usize, usize), String> {
    herbert_chat::media::target_size(h, w, v.patch, v.max_soft * v.pool * v.pool, v.pool)
}

/// The soft tokens of an `h × w` image: its patches once resized, over the pooling window.
pub fn soft_tokens(h: usize, w: usize, v: Vision) -> Result<usize, String> {
    let (th, tw) = target_size(h, w, v)?;
    Ok((th / v.patch) * (tw / v.patch) / (v.pool * v.pool))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GEMMA: Vision = Vision {
        patch: 16,
        pool: 3,
        max_soft: 280,
    };

    #[test]
    fn soft_tokens_stay_under_the_budget() {
        for (h, w) in [
            (480, 640),
            (1080, 1920),
            (512, 512),
            (4000, 3000),
            (100, 2000),
            (2000, 60),
        ] {
            let n = soft_tokens(h, w, GEMMA).unwrap();
            assert!(n > 0 && n <= GEMMA.max_soft, "{h}×{w}: {n}");
        }
        // 768 × 768: 48 × 48 patches, the 3 × 3 pooling: 256 soft tokens.
        assert_eq!(target_size(768, 768, GEMMA).unwrap(), (768, 768));
        assert_eq!(soft_tokens(768, 768, GEMMA).unwrap(), 256);
    }

    #[test]
    fn a_png_part_is_decoded_and_told_apart() {
        let mut png = Vec::new();
        let img =
            image::RgbImage::from_fn(4, 3, |x, y| image::Rgb([x as u8 * 60, y as u8 * 80, 7]));
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let url = format!("data:image/png;base64,{}", base64::encode(&png));
        let messages = json!([
            {"role": "system", "content": "Tu es utile."},
            {"role": "user", "content": [{"type": "text", "text": "Quoi ?"}, {"type": "image_url", "image_url": {"url": url}}]},
        ]);
        let got = collect(&messages).unwrap();
        assert_eq!(
            (got.len(), got[0].h, got[0].w, got[0].rgb.len()),
            (1, 3, 4, 36)
        );
        assert_eq!(&got[0].rgb[3..6], &[60, 0, 7]);
        let other = image::RgbImage::from_fn(4, 3, |x, _| image::Rgb([x as u8, 0, 0]));
        let mut png2 = Vec::new();
        other
            .write_to(
                &mut std::io::Cursor::new(&mut png2),
                image::ImageFormat::Png,
            )
            .unwrap();
        assert_ne!(
            decode(&format!("data:image/png;base64,{}", base64::encode(&png2)))
                .unwrap()
                .digest,
            got[0].digest
        );
        assert!(decode("https://example.com/a.png").is_err());
    }
}
