//! The recordings of a request: the `input_audio` parts of the messages (OpenAI's shape, `{"type":
//! "input_audio", "input_audio": {"data": base64, "format": "wav"}}`), the WAV read (PCM 8/16/24/32
//! bits or float 32, the channels averaged), resampled to the engine's rate as librosa's `soxr_hq`
//! would (what a client of transformers reads them with) and cut to what the processor keeps; the
//! soft tokens each takes as Gemma 4's processor counts them ([`Audio`]).
//!
//! To transcribe, a client puts the instruction before the recording (Google's own prompt for Gemma
//! 4 does), or says it in a system message: with the recording first and the instruction after it,
//! Gemma 4 12B takes a dictation that sounds like a request ("look on X for…") for the request and
//! answers it. Measured on eight French dictations, with the prompts in French as a user of them
//! writes them, the 12B's word error rate goes from 20.4 % (recording, then "Transcris mot pour
//! mot…", transcribe word for word) to 8.7–11.7 % with the instruction first, a system message, or
//! the recording framed by "Début de l'enregistrement :" … "Fin de l'enregistrement." (start and
//! end of the recording); E4B stays at 9.3–9.9 % whatever the wording; transformers 5.22.1 answers
//! the same dictation in the same wordings (measured on 2026-11-11).

use crate::backend::Audio;
use serde_json::Value;

/// One recording of a request, at the engine's rate.
pub struct Recording {
    /// Mono, at [`Audio::rate`], cut to [`Audio::max_samples`].
    pub wave: Vec<f32>,
    /// The file's own rate and length in seconds, before the cut.
    pub rate: usize,
    pub seconds: f64,
    /// FNV-1a over the samples fed: what tells two recordings apart in the tokens the server keeps
    /// (`worker.rs`).
    pub digest: u64,
}

/// The recordings of `messages`, in the order the templates write their placeholders: message by
/// message, the parts of each in order (a part `{"type": "input_audio", "input_audio": {"data": …,
/// "format": "wav"}}`, or `"type": "audio"` with the same object under `audio`). `a`: the engine's
/// tower; a recording without one is refused.
pub fn collect(messages: &Value, a: Option<Audio>) -> Result<Vec<Recording>, String> {
    let mut out = Vec::new();
    for m in messages.as_array().into_iter().flatten() {
        for part in m["content"].as_array().into_iter().flatten() {
            if !matches!(part["type"].as_str(), Some("input_audio" | "audio")) {
                continue;
            }
            let at = out.len();
            let a = a.ok_or("this model hears no recording")?;
            let obj = if part["input_audio"].is_object() {
                &part["input_audio"]
            } else {
                &part["audio"]
            };
            let format = obj["format"].as_str().unwrap_or("wav");
            if format != "wav" {
                return Err(format!(
                    "recording {at}: format {format:?}, only wav is read"
                ));
            }
            let data = obj["data"]
                .as_str()
                .ok_or(format!("recording {at}: no data in the part"))?;
            if data.len() / 4 * 3 > crate::image::MAX_BYTES {
                return Err(format!(
                    "recording {at}: more than {} MiB",
                    crate::image::MAX_BYTES >> 20
                ));
            }
            let bytes =
                base64::decode(data.trim()).map_err(|e| format!("recording {at}: base64: {e}"))?;
            out.push(decode(&bytes, a).map_err(|e| format!("recording {at}: {e}"))?);
        }
    }
    Ok(out)
}

/// A WAV file's bytes → the recording the engine takes: mono, at `a.rate`, cut to `a.max_samples`.
pub fn decode(bytes: &[u8], a: Audio) -> Result<Recording, String> {
    let (wave, rate) = read_wav(bytes)?;
    let seconds = wave.len() as f64 / rate as f64;
    // Cut before the resampling what it does not need (a long recording, once cut, costs no more
    // than 30 s).
    let keep = ((a.max_samples as u64 * rate as u64).div_ceil(a.rate as u64) as usize + 256)
        .min(wave.len());
    let mut wave = resample_kaiser_best(&wave[..keep], rate, a.rate)?;
    wave.truncate(a.max_samples);
    if a.soft_tokens(wave.len()) == 0 {
        return Err(format!(
            "{:.1} ms of audio: too short for one soft token",
            seconds * 1e3
        ));
    }
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for b in (wave.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain(wave.iter().flat_map(|s| s.to_le_bytes()))
    {
        digest = (digest ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(Recording {
        wave,
        rate,
        seconds,
        digest,
    })
}

/// The samples of a RIFF/WAVE file (PCM 8/16/24/32 bits or float 32), the channels averaged, and
/// its rate.
pub fn read_wav(b: &[u8]) -> Result<(Vec<f32>, usize), String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let (mut at, mut fmt, mut data) = (12usize, None, None);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        let body = at + 8;
        let end = body.saturating_add(len).min(b.len());
        if id == b"fmt " && end - body >= 16 {
            let u16at = |o: usize| u16::from_le_bytes([b[body + o], b[body + o + 1]]);
            let mut tag = u16at(0);
            if tag == 0xFFFE && end - body >= 26 {
                tag = u16at(24); // WAVE_FORMAT_EXTENSIBLE: the sub-format's leading tag
            }
            let rate =
                u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]) as usize;
            fmt = Some((tag, u16at(2) as usize, rate, u16at(14) as usize));
        } else if id == b"data" {
            data = Some(&b[body..end]);
        }
        at = body.saturating_add(len).saturating_add(len & 1);
    }
    let (tag, channels, rate, bits) = fmt.ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    if channels == 0 || rate == 0 || rate > 384_000 {
        return Err(format!("{channels} channels at {rate} Hz"));
    }
    let width = bits / 8;
    let one: fn(&[u8]) -> f32 = match (tag, bits) {
        (1, 8) => |s| (s[0] as f32 - 128.0) / 128.0,
        (1, 16) => |s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
        (1, 24) => |s| (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
        (1, 32) => |s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        _ => {
            return Err(format!(
                "format {tag} at {bits} bits is not read (PCM 8/16/24/32 bits or float 32)"
            ))
        }
    };
    let frame = width * channels;
    let out = data
        .chunks_exact(frame)
        .map(|f| {
            (0..channels)
                .map(|c| one(&f[c * width..(c + 1) * width]))
                .sum::<f32>()
                / channels as f32
        })
        .collect();
    Ok((out, rate))
}

pub use herbert_chat::media::resample_kaiser_best;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::AudioFraming;
    use serde_json::json;

    /// A WAV file of 16-bit PCM: `channels` interleaved, `rate` Hz.
    pub(crate) fn wav16(samples: &[i16], channels: u16, rate: u32) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        b.extend_from_slice(&(channels * 2).to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&data);
        b
    }

    /// The processor's counts (`_compute_audio_num_tokens`): 1 s at 16 kHz is 99 frames, 50 then 25
    /// soft tokens; 30 s is 750, and nothing past it; 12B's chunks of 640 samples.
    #[test]
    fn soft_tokens_as_the_processor_counts_them() {
        let a = Audio::GEMMA4;
        assert_eq!(a.soft_tokens(16_000), 25);
        assert_eq!(a.soft_tokens(480_000), 750);
        assert_eq!(a.soft_tokens(10 * 480_000), 750);
        assert_eq!(a.soft_tokens(160), 0);
        assert_eq!(a.soft_tokens(161), 1);
        // 2.5 s: (40000 + 160 − 321) / 160 + 1 = 249 frames, 125, 63.
        assert_eq!(a.soft_tokens(40_000), 63);
        let u = Audio {
            framing: AudioFraming::Chunks(640),
            ..Audio::GEMMA4
        };
        assert_eq!(
            (
                u.soft_tokens(640),
                u.soft_tokens(641),
                u.soft_tokens(16_000),
                u.soft_tokens(10 * 480_000)
            ),
            (1, 2, 25, 750)
        );
    }

    #[test]
    fn a_wav_part_is_read_resampled_and_told_apart() {
        // 0.5 s of a 440 Hz tone at 48 kHz, stereo with the second channel silent: mono at half the
        // amplitude, 16 kHz.
        let n = 24_000;
        let tone: Vec<i16> = (0..n)
            .flat_map(|i| {
                [
                    ((i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 16_000.0) as i16,
                    0,
                ]
            })
            .collect();
        let url = base64::encode(wav16(&tone, 2, 48_000));
        let messages = json!([
            {"role": "user", "content": [{"type": "text", "text": "Quoi ?"}, {"type": "input_audio", "input_audio": {"data": url, "format": "wav"}}]},
        ]);
        let got = collect(&messages, Some(Audio::GEMMA4)).unwrap();
        assert_eq!(
            (got.len(), got[0].wave.len(), got[0].rate),
            (1, 8_000, 48_000)
        );
        assert!((got[0].seconds - 0.5).abs() < 1e-9);
        // The tone kept, at the new rate: its peak near 16000 / 32768 / 2 away from the edges.
        let peak = got[0].wave[1000..7000]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.244).abs() < 0.01, "{peak}");
        // At 16 kHz already: the samples as they are.
        let same: Vec<i16> = (0..16_000).map(|i| (i % 200) as i16 * 10).collect();
        let r = decode(&wav16(&same, 1, 16_000), Audio::GEMMA4).unwrap();
        assert_eq!(r.wave.len(), 16_000);
        assert_eq!(r.wave[13], 130.0 / 32768.0);
        assert_ne!(r.digest, got[0].digest);
        // 40 s: cut to 30.
        let long = decode(&wav16(&vec![1i16; 640_000], 1, 16_000), Audio::GEMMA4).unwrap();
        assert_eq!(long.wave.len(), 480_000);
        assert!(collect(&messages, None).is_err());
        assert!(decode(b"RIFF....WAVE", Audio::GEMMA4).is_err());
        assert!(decode(&wav16(&[1, 2, 3], 1, 16_000), Audio::GEMMA4).is_err());
    }

    /// A rate whose ratio to 16 kHz does not reduce is refused at once, not resampled with a table
    /// of gigabytes; the usual rates stay under the bound.
    #[test]
    fn odd_rates_are_refused() {
        let Err(e) = decode(&wav16(&vec![0i16; 44_101], 1, 44_101), Audio::GEMMA4) else {
            panic!("44 101 Hz resampled");
        };
        assert!(e.contains("44101 Hz"), "{e}");
        for rate in [
            8_000, 11_025, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 192_000,
        ] {
            assert!(
                resample_kaiser_best(&[0.0; 64], rate, 16_000).is_ok(),
                "{rate}"
            );
        }
    }
}
