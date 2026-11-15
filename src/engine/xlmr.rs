//! XLM-RoBERTa on the CPU: BAAI/bge-m3 (`XLMRobertaModel`, its dense vector: the first token's
//! hidden state, 1024 dimensions) on `/v1/embeddings`, BAAI/bge-reranker-v2-m3
//! (`XLMRobertaForSequenceClassification`, one logit) on `/v1/rerank`, as transformers'
//! `modeling_xlm_roberta.py` computes them; the encoder in one generated program (`jit/schedule/
//! encoder.rs`), one input at a time, all its rows in one block.
//!
//! The host copies the input's rows of the word table (250 002 rows, left in the checkpoint's
//! mapping) in bf16; the program adds the positions (counted from `padding_idx + 1`, as
//! `create_position_ids_from_input_ids` does for an unpadded input) and the token type, normalizes,
//! then runs the layers (post-norm, exact GeLU). The reranker's head (`RobertaClassificationHead`:
//! `out_proj(tanh(dense(h[0])))`) is a second program over the CLS row, its logit in
//! f32 (sentence-transformers' `CrossEncoder.predict` gives its sigmoid).
//!
//! The projections are the text's GEMMs: bf16 by default, 4-bit under `--q4` (int8 under `--int8`).
//! The sparse and ColBERT heads of bge-m3 (`sparse_linear.pt`, `colbert_linear.pt`) are not read.

use crate::core::avec::AVec;
use crate::core::error::{HerbertError, Result};
use crate::core::hugepages::HugeVec;
use crate::engine::tensors::{from_bf16, to_bf16, Tensors};
use crate::engine::tuning::Tuning;
use crate::jit::execution::build::Mounted;
use crate::jit::model::apart::Apart;
use crate::jit::schedule::encoder::{
    lower_xlmr, lower_xlmr_head, XlmrBuffers, XlmrLayer, XlmrShape, XlmrWeights,
};
use crate::jit::Standalone;
use crate::model::weight::ROWS_INTERLEAVED;
use crate::model::PackedBf16R16Weight;
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

/// What the checkpoint is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `XLMRobertaModel`: the first token's hidden state (bge-m3's dense vector).
    Embedding,
    /// `XLMRobertaForSequenceClassification` with one label: a logit (a cross-encoder reranker).
    Reranker,
}

/// The fields of `config.json` the encoder reads.
#[derive(Clone, Debug)]
pub struct XlmrConfig {
    pub kind: Kind,
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub inter: usize,
    pub eps: f32,
    pub max_positions: usize,
    pub padding_idx: usize,
    pub vocab: usize,
}

/// Whether `dir`'s `config.json` is an XLM-RoBERTa (`model_type` `xlm-roberta`, or an `XLMRoberta…`
/// architecture).
pub fn is_xlmr(dir: &Path) -> bool {
    let Ok(v) = std::fs::read(dir.join("config.json"))
        .map_err(|_| ())
        .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|_| ()))
    else {
        return false;
    };
    v["model_type"].as_str() == Some("xlm-roberta")
        || v["architectures"].as_array().is_some_and(|a| {
            a.iter()
                .any(|x| x.as_str().is_some_and(|s| s.starts_with("XLMRoberta")))
        })
}

impl XlmrConfig {
    pub fn from_dir(dir: &Path) -> Result<XlmrConfig> {
        let v: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let n = |key: &str| {
            v[key]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| HerbertError::Config(format!("config.json: no {key}")))
        };
        let arch: Vec<&str> = v["architectures"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let kind = if arch.contains(&"XLMRobertaForSequenceClassification") {
            let labels = v["id2label"]
                .as_object()
                .map_or(v["num_labels"].as_u64().unwrap_or(2) as usize, |o| o.len());
            if labels != 1 {
                return Err(HerbertError::Config(format!(
                    "a sequence classifier of {labels} labels: a reranker has one"
                )));
            }
            Kind::Reranker
        } else if arch.is_empty() || arch.contains(&"XLMRobertaModel") {
            Kind::Embedding
        } else {
            return Err(HerbertError::Config(format!("architectures {arch:?}: XLMRobertaModel (embedding) or XLMRobertaForSequenceClassification (reranker)")));
        };
        if let Some(a) = v["hidden_act"].as_str().filter(|a| *a != "gelu") {
            return Err(HerbertError::Config(format!(
                "hidden_act {a:?}: the encoder computes gelu"
            )));
        }
        if let Some(p) = v["position_embedding_type"]
            .as_str()
            .filter(|p| *p != "absolute")
        {
            return Err(HerbertError::Config(format!(
                "position_embedding_type {p:?}: the encoder computes absolute"
            )));
        }
        Ok(XlmrConfig {
            kind,
            hidden: n("hidden_size")?,
            heads: n("num_attention_heads")?,
            layers: n("num_hidden_layers")?,
            inter: n("intermediate_size")?,
            eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            max_positions: n("max_position_embeddings")?,
            padding_idx: v["pad_token_id"].as_u64().unwrap_or(1) as usize,
            vocab: n("vocab_size")?,
        })
    }

    /// The tokens of one input at most: the positions past the padding's.
    pub fn max_tokens(&self) -> usize {
        self.max_positions - self.padding_idx - 1
    }
}

struct OwnedLayer {
    linears: Vec<PackedBf16R16Weight>,
    biases: Vec<AVec<u16>>,
    norms: [AVec<u16>; 2],
}

/// The reranker's head: its program and what it reads.
struct Head {
    program: Mounted,
    #[allow(dead_code)]
    dense: PackedBf16R16Weight,
    #[allow(dead_code)]
    dense_bias: AVec<u16>,
    #[allow(dead_code)]
    out_proj: PackedBf16R16Weight,
    #[allow(dead_code)]
    t: AVec<u16>,
    logits: AVec<f32>,
    #[allow(dead_code)]
    packed: AVec<u16>,
    out_bias: f32,
    /// The CLS row the program reads, copied from the input's first row.
    cls: AVec<u16>,
}

pub struct XlmrEngine {
    pub c: XlmrConfig,
    shape: XlmrShape,
    // The programs hold the addresses of everything below them: declared first, dropped first.
    program: Mounted,
    head: Option<Head>,
    buffers: Box<XlmrBuffers>,
    #[allow(dead_code)]
    layers: Vec<OwnedLayer>,
    /// `position_embeddings` from row `padding_idx + 1`, `[rows][hidden]`: copied into each
    /// block's position rows.
    positions: AVec<u16>,
    #[allow(dead_code)]
    token_type: AVec<u16>,
    #[allow(dead_code)]
    norm: AVec<u16>,
    t: Tensors,
    exec: Standalone,
    tokens: Vec<u32>,
    /// `HERBERT_ENCODER_TIMES=1`: the times of each input on stderr.
    times: bool,
    /// `HERBERT_ENCODER_PROFILE=1`: the encoder's time by operation on stderr.
    profile: bool,
    /// The inputs of one block at most (`HERBERT_ENCODER_BATCH`, 256 by default; 1: one input at a
    /// time).
    pub batch: usize,
    pub load_seconds: f64,
    /// `--q4` (or `--int8`) took the projections.
    pub quantized: bool,
}

/// Where the loader looks for the importance of the Q4 projections, under the model's directory: a
/// subdirectory, so that no loader that maps every `*.safetensors` of the directory takes it for a
/// shard.
pub const IMPORTANCE_FILE: &str = "herbert/xlmr-q4-importance.safetensors";

/// The importance of each input column of the Q4 projections, by weight name: a safetensors file
/// (format `herbert-xlmr-q4-importance/1` in its metadata) holding, for every projection the
/// engine quantizes to Q4, one f32 tensor of its `k` input columns, E[x²] of that input over
/// calibration texts with the model in f32 (the script that made it is not in this repository): the
/// Q4 scales of each group of 32 weights are chosen to minimize the error weighted by it
/// (`jit::int8::pack_q4_weighted`) instead of on `Q4_0`'s grid. Shared bench, Ryzen 9 9900X:
/// bge-reranker-v2-m3 mean logit gap to f32 0.4372 → 0.3809, Spearman 0.9627 → 0.9687; bge-m3 mean
/// cosine 0.9505 → 0.9633 (README).
struct Importance {
    path: std::path::PathBuf,
    by_name: std::collections::HashMap<String, Vec<f32>>,
}

impl Importance {
    /// `HERBERT_XLMR_Q4_IMP`: unset or `1`, `dir/`[`IMPORTANCE_FILE`]; `0`, none (`Q4_0`'s grid,
    /// the bytes of before); otherwise the path of the file. Absent, unreadable, or not this
    /// model's (format, sizes, a projection missing, a value negative or not finite): one line
    /// on stderr and none.
    fn load(dir: &Path, c: &XlmrConfig) -> Option<Importance> {
        let knob = std::env::var("HERBERT_XLMR_Q4_IMP").unwrap_or_default();
        if knob == "0" {
            eprintln!("[xlmr] HERBERT_XLMR_Q4_IMP=0: Q4 on Q4_0's grid");
            return None;
        }
        let path = if knob.is_empty() || knob == "1" {
            dir.join(IMPORTANCE_FILE)
        } else {
            std::path::PathBuf::from(&knob)
        };
        match Self::read(&path, c) {
            Ok(i) => Some(i),
            Err(e) => {
                eprintln!(
                    "[xlmr] no input importance ({}: {e}): Q4 on Q4_0's grid",
                    path.display()
                );
                None
            }
        }
    }

    fn read(path: &Path, c: &XlmrConfig) -> std::result::Result<Importance, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let (_, meta) =
            safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| e.to_string())?;
        let meta = meta.metadata().clone().unwrap_or_default();
        let get = |k: &str| meta.get(k).map(String::as_str).unwrap_or("");
        if get("format") != "herbert-xlmr-q4-importance/1" {
            return Err(format!("format {:?}", get("format")));
        }
        let (hidden, layers, inter) = (
            c.hidden.to_string(),
            c.layers.to_string(),
            c.inter.to_string(),
        );
        if get("hidden") != hidden || get("layers") != layers || get("intermediate") != inter {
            return Err(format!(
                "made for hidden {}, {} layers, MLP {}",
                get("hidden"),
                get("layers"),
                get("intermediate")
            ));
        }
        let kind = if c.kind == Kind::Reranker {
            "reranker"
        } else {
            "embedding"
        };
        if get("kind") != kind {
            return Err(format!(
                "made for a {:?} model, this one is a {kind}",
                get("kind")
            ));
        }
        let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
        let mut by_name = std::collections::HashMap::new();
        let (h, inter) = (c.hidden, c.inter);
        let mut want: Vec<(String, usize)> = Vec::new();
        for l in 0..c.layers {
            for (name, k) in [
                ("attention.self.query", h),
                ("attention.self.key", h),
                ("attention.self.value", h),
                ("attention.output.dense", h),
                ("intermediate.dense", h),
                ("output.dense", inter),
            ] {
                want.push((format!("encoder.layer.{l}.{name}.weight"), k));
            }
        }
        if c.kind == Kind::Reranker {
            want.push(("classifier.dense.weight".into(), h));
        }
        for (name, k) in want {
            let view = st.tensor(&name).map_err(|_| format!("no {name}"))?;
            let v = crate::model::loader::load_f32_from_view(&view, &name)
                .map_err(|e| e.to_string())?;
            if v.len() != k || v.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(format!(
                    "{name}: {} values, {k} expected, all finite and ≥ 0",
                    v.len()
                ));
            }
            by_name.insert(name, v);
        }
        Ok(Importance {
            path: path.to_path_buf(),
            by_name,
        })
    }

    /// The importance of the `k` input columns of weight `name`.
    fn of(&self, name: &str, k: usize) -> Option<&[f32]> {
        self.by_name
            .get(name)
            .map(Vec::as_slice)
            .filter(|v| v.len() == k)
    }
}

/// A LayerNorm's `g` then `b`, bf16.
fn ln_weights(t: &Tensors, prefix: &str, h: usize) -> Result<AVec<u16>> {
    let mut v = t.bf16(&format!("{prefix}.weight"), h)?;
    v.extend(t.bf16(&format!("{prefix}.bias"), h)?);
    Ok(AVec::from_vec(v))
}

impl XlmrEngine {
    /// Read, pack and mount the encoder of `dir` for inputs of `capacity` tokens at most, on
    /// `threads` workers, with the tuning's knobs (`--q4`, `--int8`).
    pub fn load(
        dir: &Path,
        threads: usize,
        capacity: usize,
        tuning: &Tuning,
    ) -> Result<XlmrEngine> {
        if !crate::jit::engine_supported() || cfg!(target_arch = "riscv64") {
            return Err(HerbertError::Backend(
                "XLM-RoBERTa: the encoder needs x86-64 (AVX2 or AVX-512) or AArch64".into(),
            ));
        }
        crate::pool::power::opt_out_of_power_throttling();
        tuning.install()?;
        if threads > 0 {
            crate::pool::thread_pool::configure_global_pool(threads)
                .map_err(|e| HerbertError::Backend(format!("thread pool: {e}")))?;
        }
        let t0 = Instant::now();
        let dir = crate::model::loader::resolve_model_dir(dir)?;
        let c = XlmrConfig::from_dir(&dir)?;
        let (h, nh, inter) = (c.hidden, c.heads, c.inter);
        if nh == 0 || h % nh != 0 || (h / nh) % 32 != 0 || h % 32 != 0 || inter % 32 != 0 {
            return Err(HerbertError::Config(format!(
                "hidden {h}, {nh} heads, MLP {inter}: not served by this encoder"
            )));
        }
        let t = Tensors::open(&dir, &["roberta."])?;
        let q4_grid8 = crate::jit::int8::gemm(h, h)
            && crate::jit::int8::q4_for(h)
            && crate::jit::knobs::knobs().neon.q4_grid8;
        let importance = if q4_grid8 {
            Importance::load(&dir, &c)
        } else {
            None
        };
        // The pool only when the search runs: created here, before the grid's serial packing, it
        // made that packing 4.83–4.90 s instead of 4.17–4.19 (Ryzen 9 9900X, measured; the
        // cause not measured).
        let pool_workers = if importance.is_some() {
            crate::pool::thread_pool::global_pool().num_workers()
        } else {
            1
        };
        let mut hist = [0u64; crate::jit::int8::Q4_WEIGHTED_TRIES];
        let mut weighted = 0usize;
        let mut pack = |name: &str, n: usize, k: usize| -> Result<PackedBf16R16Weight> {
            let data = t.bf16(name, n * k)?;
            match importance.as_ref().and_then(|i| i.of(name, k)) {
                Some(imp) if crate::jit::int8::gemm(n, k) && crate::jit::int8::q4_for(n) => {
                    let (bytes, won) =
                        crate::jit::int8::pack_q4_weighted(&data, n, k, imp, pool_workers);
                    hist.iter_mut().zip(won).for_each(|(a, b)| *a += b);
                    weighted += 1;
                    let words: Vec<u16> = bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    Ok(PackedBf16R16Weight {
                        data_r16: HugeVec::from_vec(words),
                        n,
                        k,
                        k_even: k,
                        n_blocks: n.div_ceil(ROWS_INTERLEAVED),
                    })
                }
                _ => PackedBf16R16Weight::pack_from_row_major_r16_only(&data, n, k),
            }
        };
        let mut layers = Vec::with_capacity(c.layers);
        herbert_chat::progress::stage("weights", 0.0, 0.95);
        for l in 0..c.layers {
            let p = format!("encoder.layer.{l}.");
            let mut linears = Vec::with_capacity(6);
            let mut biases = Vec::with_capacity(6);
            for (name, n, k) in [
                ("attention.self.query", h, h),
                ("attention.self.key", h, h),
                ("attention.self.value", h, h),
                ("attention.output.dense", h, h),
                ("intermediate.dense", inter, h),
                ("output.dense", h, inter),
            ] {
                linears.push(pack(&format!("{p}{name}.weight"), n, k)?);
                biases.push(AVec::from_vec(t.bf16(&format!("{p}{name}.bias"), n)?));
            }
            let norms = [
                ln_weights(&t, &format!("{p}attention.output.LayerNorm"), h)?,
                ln_weights(&t, &format!("{p}output.LayerNorm"), h)?,
            ];
            layers.push(OwnedLayer {
                linears,
                biases,
                norms,
            });
            herbert_chat::progress::step(l + 1, c.layers);
        }
        let rows = capacity.clamp(1, c.max_tokens());
        let start = c.padding_idx + 1;
        let positions = AVec::from_vec(
            t.bf16("embeddings.position_embeddings.weight", c.max_positions * h)?
                [start * h..(start + rows) * h]
                .to_vec(),
        );
        let token_type = AVec::from_vec(
            t.f32_range("embeddings.token_type_embeddings.weight", 0, h)?
                .into_iter()
                .map(to_bf16)
                .collect(),
        );
        let norm = ln_weights(&t, "embeddings.LayerNorm", h)?;
        let quantized = crate::jit::int8::gemm(h, h);
        let shape = XlmrShape {
            hidden: h,
            heads: nh,
            intermediate: inter,
            layers: c.layers,
            eps: c.eps,
            rows,
        };
        let workers = crate::pool::thread_pool::global_pool().num_workers();
        let mut buffers = Box::new(XlmrBuffers::new(&shape, workers));
        let exec = Standalone::new(tuning.executor_options());
        // `HERBERT_ENCODER_PROFILE=1` (x86-64): the program stamps its nodes, their times summed by
        // operation per input.
        let profile =
            cfg!(target_arch = "x86_64") && std::env::var_os("HERBERT_ENCODER_PROFILE").is_some();
        let t_read = t0.elapsed().as_secs_f64();
        let program = {
            let weights = XlmrWeights {
                token_type: &token_type,
                norm: &norm,
                layers: layers
                    .iter()
                    .map(|l| XlmrLayer {
                        linears: std::array::from_fn(|i| &l.linears[i].data_r16[..]),
                        biases: std::array::from_fn(|i| &l.biases[i][..]),
                        norms: [&l.norms[0], &l.norms[1]],
                    })
                    .collect(),
            };
            lower_xlmr(
                &shape,
                &weights,
                &mut buffers,
                workers,
                exec.apart_spin(),
                profile,
            )
            .map_err(|e| HerbertError::Backend(format!("xlmr: {e}")))?
        };
        let head = match c.kind {
            Kind::Embedding => None,
            Kind::Reranker => {
                let dense = pack("classifier.dense.weight", h, h)?;
                let dense_bias = AVec::from_vec(t.bf16("classifier.dense.bias", h)?);
                let mut out = t.bf16("classifier.out_proj.weight", h)?;
                out.resize(16 * h, 0);
                let out_proj = PackedBf16R16Weight::pack_from_row_major_r16_only(&out, 16, h)?;
                let out_bias = t.f32("classifier.out_proj.bias", 1)?[0];
                let mut tb = AVec::zeroed(h);
                let mut logits = AVec::zeroed(16);
                let mut packed =
                    AVec::zeroed((crate::jit::int8::a_bytes(1, h) + 64 * h).div_ceil(2));
                let mut cls = AVec::zeroed(h);
                let program = lower_xlmr_head(
                    h,
                    cls.as_mut_ptr(),
                    &dense.data_r16,
                    &dense_bias,
                    &out_proj.data_r16,
                    &mut tb,
                    &mut logits,
                    &mut packed,
                    workers,
                    exec.apart_spin(),
                )
                .map_err(|e| HerbertError::Backend(format!("xlmr head: {e}")))?;
                Some(Head {
                    program,
                    dense,
                    dense_bias,
                    out_proj,
                    t: tb,
                    logits,
                    packed,
                    out_bias,
                    cls,
                })
            }
        };
        if let Some(i) = &importance {
            eprintln!(
                "[xlmr] Q4 scales weighted by input importance ({}): {weighted} projections",
                i.path.display()
            );
            if std::env::var_os("HERBERT_XLMR_Q4_IMP_STATS").is_some() {
                let won: Vec<String> = hist
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c > 0)
                    .map(|(i, c)| {
                        format!(
                            "s={}{}:{c}",
                            i as i32 / 2 - 9,
                            if i % 2 == 1 { "r2" } else { "r1" }
                        )
                    })
                    .collect();
                eprintln!("[xlmr] tries won: {}", won.join(" "));
            }
        }
        let load_seconds = t0.elapsed().as_secs_f64();
        eprintln!(
            "[xlmr] {:?}: {} layers, hidden {h}, {nh} heads of {}, MLP {inter}; {} projections; read and packed in {t_read:.2} s, mounted in {:.2} s; {:.0} MiB of buffers for {rows} tokens; {} bytes of code; the word table ({} rows) stays mapped",
            c.kind,
            c.layers,
            h / nh,
            if quantized { if crate::jit::int8::q4_for(h) { "Q4" } else { "int8" } } else { "bf16" },
            load_seconds - t_read,
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            program.program.code().len(),
            c.vocab
        );
        Ok(XlmrEngine {
            c,
            shape,
            program,
            head,
            buffers,
            layers,
            positions,
            token_type,
            norm,
            t,
            exec,
            tokens: vec![0; rows],
            times: std::env::var_os("HERBERT_ENCODER_TIMES").is_some(),
            profile,
            batch: std::env::var("HERBERT_ENCODER_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            load_seconds,
            quantized,
        })
    }

    /// The tokens of one input at most.
    pub fn capacity(&self) -> usize {
        self.shape.rows
    }

    /// The inputs `inputs` (the tokens of each, special tokens included) end to end in one block,
    /// the last layer's rows left in the buffer; the first row of each input. Each row attends
    /// to its own input's rows only.
    fn encode(&mut self, inputs: &[&[u32]]) -> Result<Vec<usize>> {
        let t0 = Instant::now();
        let h = self.c.hidden;
        let m: usize = inputs.iter().map(|i| i.len()).sum();
        if inputs.iter().any(|i| i.is_empty()) || m > self.shape.rows {
            return Err(HerbertError::Backend(format!(
                "{m} tokens in {} inputs: between 1 and {} each, {} in all",
                inputs.len(),
                self.shape.rows,
                self.shape.rows
            )));
        }
        let align = encoder_align();
        let m = inputs
            .iter()
            .fold(0, |r: usize, i| r.next_multiple_of(align) + i.len());
        if m > self.shape.rows {
            return Err(HerbertError::Backend(format!(
                "{m} rows for {} inputs aligned on {align}: {} at most",
                inputs.len(),
                self.shape.rows
            )));
        }
        let mut starts = Vec::with_capacity(inputs.len());
        let mut r = 0usize;
        for ids in inputs {
            // The rows up to the input's aligned start: zeros, in the previous input's segment
            // (keys of no other input).
            let to = r.next_multiple_of(align);
            if to > r {
                let prev = (
                    self.buffers.segments[2 * r - 2],
                    self.buffers.segments[2 * r - 1],
                );
                self.buffers.input[r * h..to * h].fill(0);
                self.buffers.positions[r * h..to * h]
                    .copy_from_slice(&self.positions[..(to - r) * h]);
                for p in r..to {
                    self.buffers.segments[2 * p] = prev.0;
                    self.buffers.segments[2 * p + 1] = prev.1;
                }
                r = to;
            }
            starts.push(r);
            let (start, end) = (r as u32, (r + ids.len()) as u32);
            for (p, &id) in ids.iter().enumerate() {
                if id as usize >= self.c.vocab {
                    return Err(HerbertError::Backend(format!(
                        "token {id} past the vocabulary of {}",
                        self.c.vocab
                    )));
                }
                let row =
                    self.t
                        .f32_range("embeddings.word_embeddings.weight", id as usize * h, h)?;
                for (d, v) in self.buffers.input[r * h..(r + 1) * h].iter_mut().zip(row) {
                    *d = to_bf16(v);
                }
                self.buffers.positions[r * h..(r + 1) * h]
                    .copy_from_slice(&self.positions[p * h..(p + 1) * h]);
                self.buffers.segments[2 * r] = start;
                self.buffers.segments[2 * r + 1] = end;
                r += 1;
            }
        }
        if self.profile {
            self.exec.reserve_apart_trace(&self.program)?;
        }
        let t1 = Instant::now();
        self.exec.run_apart(&mut self.program, &self.tokens[..m])?;
        if self.profile {
            crate::engine::vision::print_profile_as(
                "xlmr",
                &self.exec,
                &self.program,
                t1.elapsed().as_secs_f64(),
            );
        }
        if self.times {
            eprintln!(
                "[xlmr] {} inputs, {m} tokens: word rows {:.2} ms, encoder {:.1} ms",
                inputs.len(),
                (t1 - t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(starts)
    }

    /// The inputs of `inputs` in blocks of at most [`XlmrEngine::capacity`] rows, in order: `f`
    /// takes each block's inputs and their first rows.
    fn blocks<T>(
        &mut self,
        inputs: &[&[u32]],
        mut f: impl FnMut(&mut XlmrEngine, Vec<usize>) -> Result<Vec<T>>,
    ) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(inputs.len());
        let (cap, most) = (self.shape.rows, self.batch.max(1));
        let mut i = 0;
        while i < inputs.len() {
            let (mut j, mut rows) = (i, 0usize);
            let align = encoder_align();
            while j < inputs.len()
                && j - i < most
                && (j == i || rows.next_multiple_of(align) + inputs[j].len() <= cap)
            {
                rows = rows.next_multiple_of(align) + inputs[j].len();
                j += 1;
            }
            let starts = self.encode(&inputs[i..j])?;
            out.extend(f(self, starts)?);
            i = j;
        }
        Ok(out)
    }

    /// The CLS row of the input starting at row `r` (bge-m3's vector, not normalized).
    fn cls_at(&self, r: usize) -> Result<Vec<f32>> {
        let h = self.c.hidden;
        let cls: Vec<f32> = self.buffers.h[r * h..(r + 1) * h]
            .iter()
            .map(|&b| from_bf16(b))
            .collect();
        if cls.iter().any(|v| !v.is_finite()) {
            return Err(HerbertError::Backend("the CLS row is not finite".into()));
        }
        Ok(cls)
    }

    /// The first token's hidden state (bge-m3's vector, not normalized).
    pub fn encode_cls(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.encode(&[ids])?;
        self.cls_at(0)
    }

    /// [`XlmrEngine::encode_cls`] of several inputs, run end to end in blocks.
    pub fn encode_cls_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        self.blocks(inputs, |e, starts| {
            starts.iter().map(|&r| e.cls_at(r)).collect()
        })
    }

    /// The embedding model's vector size, `None` for a reranker.
    pub fn embedding_dim(&self) -> Option<usize> {
        (self.c.kind == Kind::Embedding).then_some(self.c.hidden)
    }

    /// The reranker's logit of the input starting at row `r` of the last block.
    fn logit_at(&mut self, r: usize) -> Result<f32> {
        let h = self.c.hidden;
        let head = self
            .head
            .as_mut()
            .ok_or_else(|| HerbertError::Backend("this model is not a reranker".into()))?;
        head.cls
            .copy_from_slice(&self.buffers.h[r * h..(r + 1) * h]);
        self.exec.run_apart(&mut head.program, &self.tokens[..1])?;
        let logit = head.logits[0] + head.out_bias;
        if !logit.is_finite() {
            return Err(HerbertError::Backend("the logit is not finite".into()));
        }
        Ok(logit)
    }

    /// The reranker's logit of the pair whose tokens are `ids`.
    pub fn score(&mut self, ids: &[u32]) -> Result<f32> {
        if self.head.is_none() {
            return Err(HerbertError::Backend("this model is not a reranker".into()));
        }
        self.encode(&[ids])?;
        self.logit_at(0)
    }

    /// [`XlmrEngine::score`] of several pairs, run end to end in blocks.
    pub fn score_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<f32>> {
        if self.head.is_none() {
            return Err(HerbertError::Backend("this model is not a reranker".into()));
        }
        self.blocks(inputs, |e, starts| {
            starts.iter().map(|&r| e.logit_at(r)).collect()
        })
    }
}

/// Where the inputs of a block start, end to end (`HERBERT_ENCODER_ALIGN`, rows): on a multiple of
/// sixteen rows on AArch64, whose block attention runs tiles of sixteen positions
/// (`sdpa_block.rs`). A tile across two inputs runs the row kernel, and a tile's keys start on a
/// multiple of sixteen, so an input's vector would change by a few bf16 steps with its place in the
/// block; aligned, every row of an input gets the bits it gets alone (the rows before an aligned
/// start are zeros, in the previous input's segment, and their output is never read). 1 elsewhere
/// (x86-64: the same bits wherever the input starts).
pub(crate) fn encoder_align() -> usize {
    static ALIGN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ALIGN.get_or_init(|| {
        let default = if cfg!(target_arch = "aarch64") { 16 } else { 1 };
        std::env::var("HERBERT_ENCODER_ALIGN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&a: &usize| a > 0)
            .unwrap_or(default)
    })
}
