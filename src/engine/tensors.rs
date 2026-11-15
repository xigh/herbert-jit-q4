//! The tensors of a checkpoint, mapped and read by name: what the encoders (EmbeddingGemma 2,
//! XLM-RoBERTa) take from it. Nothing is copied until a tensor is asked for; a row of a large table
//! (the word embeddings) is read alone.

use crate::core::error::{HerbertError, Result};
use crate::model::loader::{load_bf16_from_view, load_f32_from_view};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// The mapped shards and where each tensor lies in them.
pub struct Tensors {
    maps: Vec<Mmap>,
    /// Name (the prefix dropped) → shard, byte range, dtype, shape.
    at: HashMap<String, (usize, usize, usize, safetensors::Dtype, Vec<usize>)>,
}

impl Tensors {
    /// Map the shards of `dir` (`model.safetensors`, or those of `model.safetensors.index.json`);
    /// the first of `prefixes` a name starts with is dropped from it.
    pub fn open(dir: &Path, prefixes: &[&str]) -> Result<Tensors> {
        let index = dir.join("model.safetensors.index.json");
        let shards: Vec<String> = if index.exists() {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
            let mut s: Vec<String> = v["weight_map"]
                .as_object()
                .map(|o| {
                    o.values()
                        .filter_map(serde_json::Value::as_str)
                        // The file name only, as the loader reads it: never a path out of `dir`.
                        .map(|s| crate::model::loader::shard_file_name(s).to_string())
                        .collect()
                })
                .unwrap_or_default();
            s.sort();
            s.dedup();
            s
        } else {
            vec!["model.safetensors".into()]
        };
        let (mut maps, mut at) = (Vec::new(), HashMap::new());
        for shard in &shards {
            let file = std::fs::File::open(dir.join(shard))
                .map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            // SAFETY: a read-only mapping of a file nobody modifies meanwhile.
            let mapped = unsafe { Mmap::map(&file) }
                .map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            let st = SafeTensors::deserialize(&mapped[..])
                .map_err(|e| HerbertError::ModelLoad(format!("{shard}: {e}")))?;
            for (name, view) in st.tensors() {
                let start = view.data().as_ptr() as usize - mapped.as_ptr() as usize;
                let key = prefixes
                    .iter()
                    .find_map(|p| name.strip_prefix(p))
                    .unwrap_or(&name)
                    .to_string();
                at.insert(
                    key,
                    (
                        maps.len(),
                        start,
                        start + view.data().len(),
                        view.dtype(),
                        view.shape().to_vec(),
                    ),
                );
            }
            drop(st);
            maps.push(mapped);
        }
        Ok(Tensors { maps, at })
    }

    /// Whether the checkpoint holds `name`.
    pub fn has(&self, name: &str) -> bool {
        self.at.contains_key(name)
    }

    /// The shape of `name`.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.at.get(name).map(|e| e.4.as_slice())
    }

    fn view(&self, name: &str, len: usize) -> Result<safetensors::tensor::TensorView<'_>> {
        let (shard, start, end, dtype, shape) = self
            .at
            .get(name)
            .ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {name}")))?;
        let n: usize = shape.iter().product();
        if n != len {
            return Err(HerbertError::ModelLoad(format!(
                "{name}: {n} elements, {len} expected"
            )));
        }
        safetensors::tensor::TensorView::new(
            *dtype,
            shape.clone(),
            &self.maps[*shard][*start..*end],
        )
        .map_err(|e| HerbertError::ModelLoad(format!("{name}: {e}")))
    }

    /// The whole of `name` (`len` elements) in bf16 (f16 and f32 rounded to nearest even).
    pub fn bf16(&self, name: &str, len: usize) -> Result<Vec<u16>> {
        load_bf16_from_view(&self.view(name, len)?, name)
    }

    /// The whole of `name` (`len` elements) in f32 (bf16 and f16 widened exactly).
    pub fn f32(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        load_f32_from_view(&self.view(name, len)?, name)
    }

    /// Elements `from..from + len` of `name` (row-major), in f32.
    pub fn f32_range(&self, name: &str, from: usize, len: usize) -> Result<Vec<f32>> {
        let (shard, start, end, dtype, _) = self
            .at
            .get(name)
            .ok_or_else(|| HerbertError::ModelLoad(format!("missing tensor {name}")))?;
        let size = match dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
            d => return Err(HerbertError::ModelLoad(format!("{name}: dtype {d:?}"))),
        };
        if (from + len) * size > end - start {
            return Err(HerbertError::ModelLoad(format!(
                "{name}: {} elements, {} asked",
                (end - start) / size,
                from + len
            )));
        }
        let b = &self.maps[*shard][start + from * size..start + (from + len) * size];
        Ok(match dtype {
            safetensors::Dtype::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
        })
    }
}

/// `x` rounded to bf16, nearest even (the engine's 16-bit format).
pub use crate::core::tensor::f32_to_bf16 as to_bf16;

/// A bf16 value widened.
pub fn from_bf16(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}
