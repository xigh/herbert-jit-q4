//! The engine: the model mounted into the two generated regimes and driven
//! step by step.
//!
//! What enters: a model directory, the thread count, the number of positions
//! to reserve, the tuning and measurement settings.
//! What it does at `load`: reads the checkpoint into packed weights (bf16, or int8 / Q4
//! under `--int8` / `--q4`), checks the geometry, builds the `Runner` (KV cache, DeltaNet states,
//! RoPE tables, compiled kernels) and mounts both regimes — prefill and decode —
//! as two generated programs. Nothing is allocated after that.
//! What the chat calls afterwards: `prefill` (append rows to the live state),
//! `generate` (decode in one launch, one callback per token), `snapshot` and
//! `restore` (the whole state, for the disk), `write_trace` (the measurements).
//! What the server adds (`serve/llama.rs`): `prefill_until` (the prompt cut at
//! the prefill's blocks, a stop asked between them), the K/V of a prefix as
//! bytes (`kv_save`, `kv_load`, `kv_fingerprint`), the DeltaNet states apart
//! from the K/V (`recurrence_*`) and the K/V caches of other conversations
//! (`lane_*`), `touch_weights`.

use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::engine::geometry::{self, Geometry};
use crate::engine::packs::DeltaPacks;
use crate::engine::tuning::{Measure, Tuning};
use crate::jit::execution::MountOptions;
use crate::jit::{
    engine_supported, prefill_pass, Instantaneous, KvLane, LayerWeights, ModelBudget, ModelWeights,
    Recurrence, Runner, DEFAULT_MAX_BATCH,
};
use crate::model::Model;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Rows one prefill pass hands to the generated program: the pass width, capped
/// by the widest block the frame can hold.
pub fn causal_batch_width() -> usize {
    prefill_pass().min(DEFAULT_MAX_BATCH)
}

/// The loaded, mounted model. Field order matters: `layers` borrows into
/// `model` and `delta_packs`, so it is declared first and dropped first.
pub struct Engine {
    layers: Vec<LayerWeights<'static>>,
    model: Box<Model>,
    #[allow(dead_code)]
    delta_packs: Vec<Option<DeltaPacks>>,
    config: Config,
    geo: Geometry,
    runner: Runner,
    /// Positions the runner reserved: the ceiling of a conversation.
    reserved: usize,
    /// The tuning as the load left it and the workers: what the K/V of a prefix
    /// depends on besides the kernels' code (`kv_fingerprint`).
    tuning: Tuning,
    threads: usize,
    /// Wall times of the two phases, for the log.
    pub load_seconds: f64,
    pub mount_seconds: f64,
    /// The checkpoint's directory: the vision tower is read from it at the first image.
    dir: PathBuf,
    /// Gemma 4 E2B, E4B: the vision tower of the configuration (12B: its encoder-free embedder),
    /// mounted at the first image (`tower`), with the image regime of the runner reading its
    /// soft tokens (`image_tokens`: `0..IMAGE_ROWS`, their rows).
    vision: Option<crate::engine::vision::VisionConfig>,
    tower: Option<crate::engine::vision::Encoder>,
    image_tokens: Vec<u32>,
    /// Gemma 4 E2B, E4B: the audio tower of the configuration (12B: its encoder-free embedder,
    /// `audio_chunks`), mounted at the first recording (`audio_tower`), with the audio regime
    /// of the runner reading its soft tokens (`audio_tokens`: the rows of its table).
    audio: Option<crate::engine::audio::AudioConfig>,
    audio_chunks: Option<crate::engine::audio::ChunkConfig>,
    audio_tower: Option<crate::engine::audio::Hearing>,
    audio_tokens: Vec<u32>,
    /// Qwen3.5 / Qwen3.6: the vision tower of the configuration, read at the first image
    /// (`qwen_tower`), its soft tokens fed by the causal media regime of the runner (mounted
    /// on the tower's table, `qwen_table`: its address and length when mounted).
    qwen_vision: Option<crate::engine::vision_qwen::QwenVisionConfig>,
    qwen_tower: Option<crate::engine::vision_qwen::QwenTower>,
    qwen_table: (usize, usize),
    /// The multimodal RoPE of Qwen: the text after an image turns by its K/V position minus
    /// `rope_shift` (`get_rope_index`: an image of `n` soft tokens takes `max(grid_h, grid_w)`
    /// positions). The RoPE rows from `rope_clean_from` on hold the text's rotation for this
    /// offset.
    rope_shift: usize,
    rope_clean_from: usize,
    /// LFM3-VL: its vision tower (`engine/vision_lfm3.rs`), mounted at the first image, its soft
    /// tokens fed as ordinary causal positions by the runner's media regime (that of Gemma's
    /// recordings) in blocks of [`crate::engine::vision::IMAGE_ROWS`] rows (`lfm3_tokens`: the
    /// rows of its table).
    lfm3_vision: Option<crate::engine::vision_lfm3::Lfm3VisionConfig>,
    lfm3_tower: Option<crate::engine::vision_lfm3::Lfm3Tower>,
    lfm3_tokens: Vec<u32>,
    /// `HERBERT_PREFILL_PROFILE=1`: the traces of the prefill calls folded until the end of the
    /// prefill.
    prefill_profile: Option<Box<crate::engine::prefill_profile::PrefillProfile>>,
}

impl Engine {
    /// Read, check, reserve and mount. `reserve_positions` bounds the whole
    /// conversation (prompt and every generated token); it is capped by the
    /// model's RoPE tables. `tuning` and `measure` are fixed for the life of
    /// the engine: both regimes are lowered with them.
    pub fn load(
        model_dir: &Path,
        threads: usize,
        reserve_positions: usize,
        tuning: &Tuning,
        measure: &Measure,
    ) -> Result<Engine> {
        if !engine_supported() {
            return Err(HerbertError::Backend(
                if cfg!(target_arch = "aarch64") {
                    "the ISA floor is not met: NEON and LSE are required"
                } else if cfg!(target_arch = "riscv64") {
                    "the ISA floor is not met: IMA, FD, V with VLEN ≥ 256, Zba, Zbb, Zvfbfmin and Zvfbfwma are required"
                } else {
                    "the ISA floor is not met: AVX-512 (+BF16) or AVX2+FMA is required"
                }
                .to_string(),
            ));
        }
        // Before anything is read or computed: a windowless process on Windows
        // is otherwise throttled onto the LPE cores (EcoQoS), `pool::power`.
        crate::pool::power::opt_out_of_power_throttling();
        // A repository folder of the Hub's cache (`models--org--name`) stands for its snapshot,
        // for every read below.
        let model_dir = &crate::model::loader::resolve_model_dir(model_dir)?;
        let config = Config::from_file(&model_dir.join("config.json")).ok();
        let gemma_config = config.clone().filter(|c| c.gemma.is_some());
        let mut adjusted = tuning.clone();
        // Qwen2.5 (attention layers alone, hidden 1536 on the 1.5B): passes of 1024 rows instead of
        // 256, and the norms, residual additions and the chain around the attention on the
        // whole team. Ryzen 9 9900X, base 1.5B, 3 286-token prompt: 2.149 → 1.905 s (pass
        // 512 2.006, 1024 1.950, 2048 1.975; teams alone 2.108; both at 1024 1.905);
        // the decode unchanged (13.23 tokens/s). A pass or a team given on the command line is
        // kept.
        if config.as_ref().is_some_and(|c| c.model_type == "qwen2") {
            if adjusted.kernels.prefill_pass == crate::jit::schedule::PREFILL_PASS_DEFAULT {
                adjusted.kernels.prefill_pass = 1024;
            }
            adjusted.kernels.elem_team = true;
            adjusted.kernels.attn_chain_team = true;
            eprintln!(
                "[tuning] qwen2: prefill pass {}, norms and attention chain on the whole team",
                adjusted.kernels.prefill_pass
            );
        }
        let tuning = &adjusted;
        // The knobs of the emitters and of the pool, before either exists.
        tuning.install()?;
        if threads > 0 {
            crate::pool::thread_pool::configure_global_pool(threads)
                .map_err(|e| HerbertError::Backend(format!("thread pool: {e}")))?;
        }
        // Refuse before streaming what the machine cannot hold: the size of the bf16 checkpoint on
        // disk, scaled below for the int8 and 4-bit forms, is the floor of the weights.
        // The per-layer table of a Gemma 4 model stays in the mapping: only the rows of the tokens
        // are read.
        let table_bytes = gemma_config.as_ref().map_or(0, |c| {
            let g = c.gemma.as_ref().unwrap();
            (2 * g.vocab_size_per_layer_input * c.num_layers * g.pl_dim) as u64
        });
        // The int8 and 4-bit forms of the projections are smaller than the bf16 checkpoint they
        // come from (about 0.56 and 0.36 of it, scales included); the embedding table stays
        // in bf16 beside its packed form.
        let kernels = &tuning.kernels.neon;
        let factor = if kernels.w_int8 && kernels.w_q4 {
            0.36
        } else if kernels.w_int8 {
            0.56
        } else {
            1.0
        };
        let embed_bytes = gemma_config
            .as_ref()
            .map_or(0, |c| (2 * c.vocab_size * c.hidden_size) as u64);
        let matrices = crate::core::memory::checkpoint_bytes(model_dir).saturating_sub(table_bytes);
        crate::core::memory::refuse_if_it_cannot_fit(
            "weights",
            (matrices as f64 * factor) as u64 + if factor < 1.0 { embed_bytes } else { 0 },
        )?;
        let started = Instant::now();
        let (config, model) = crate::engine::loader::load_bf16_model(model_dir, true)?;
        let model = Box::new(model);
        let geo = geometry::geometry(&config, &model)?;
        let delta_packs = crate::engine::packs::delta_packs(&model)?;
        // The borrowed slices point into heap buffers that are never moved or
        // written again; `layers` is declared before them and dropped first.
        let layers: Vec<LayerWeights<'static>> = unsafe {
            std::mem::transmute(crate::engine::weights_of::layer_weights(
                &model,
                &delta_packs,
            ))
        };
        let load_seconds = started.elapsed().as_secs_f64();

        let mounted = Instant::now();
        let geometries = geometry::attention_geometries(&model.config, &model);
        let plan_kv = geometry::plan_kv(&model);
        let tables = geometry::tables_rope(&model, &geometries)?;
        // The RoPE tables bound the context: never reserve beyond them.
        let positions = tables
            .iter()
            .map(|t| {
                let half = (t.rotary_ndims / 2).max(1);
                (t.cos.len() / half).min(t.sin.len() / half)
            })
            .min()
            .unwrap_or(0);
        let reserved = reserve_positions.min(positions);
        // A mixture-of-experts model: the prefill groups the rows of a block by expert
        // (`moe_route_rows`, then `moe_combine_rows` for Gemma, `moe_qwen_out` for
        // `qwen3_5_moe`) on x86-64 and AArch64; the other cases, and `--moe-row-prefill`,
        // work one row at a time.
        let moe = model.config.gemma.as_ref().is_some_and(|g| g.moe.is_some())
            || model.config.qwen_moe.is_some();
        // The experts of a block run the int8 / 4-bit kernel (`--int8`, `--q4`; on x86-64
        // `gemm_int8.rs` — AVX-512 VNNI, AVX-VNNI or AVX2 —, `qwen3_5_moe` also with the
        // SSE4.2 kernel of `--q4-sse42`; on AArch64 `sdot`, the AArch64 floor, Gemma and
        // `qwen3_5_moe`).
        let int8_experts = if cfg!(target_arch = "x86_64") {
            if tuning.kernels.neon.w_sse42 {
                crate::core::isa::feature("avxvnni") || model.config.qwen_moe.is_some()
            } else {
                crate::jit::int8::x86_kernel_available()
            }
        } else {
            cfg!(target_arch = "aarch64")
        };
        let rows_by_expert = !tuning.moe_row_prefill && tuning.kernels.neon.w_int8 && int8_experts;
        let max_batch = if moe && !rows_by_expert {
            1
        } else {
            causal_batch_width()
        };
        // A vision tower: the buffers hold an image's soft tokens in one block (the prefill's pass
        // is unchanged).
        let vision = model
            .config
            .gemma
            .as_ref()
            .and(crate::engine::vision::VisionConfig::of(
                &crate::model::loader::resolve_model_dir(model_dir)?,
            ));
        // An audio tower: its soft tokens go in blocks of the same rows.
        let audio = model
            .config
            .gemma
            .as_ref()
            .and(crate::engine::audio::AudioConfig::of(
                &crate::model::loader::resolve_model_dir(model_dir)?,
            ));
        let audio_chunks = model
            .config
            .gemma
            .as_ref()
            .and(crate::engine::audio::ChunkConfig::of(
                &crate::model::loader::resolve_model_dir(model_dir)?,
            ));
        let qwen_vision = if model.config.gemma.is_none() {
            crate::engine::vision_qwen::QwenVisionConfig::of(
                &crate::model::loader::resolve_model_dir(model_dir)?,
            )
        } else {
            None
        };
        let lfm3_vision = crate::engine::vision_lfm3::Lfm3VisionConfig::of(
            &crate::model::loader::resolve_model_dir(model_dir)?,
        )
        .filter(|_| model.config.lfm3.is_some());
        let max_batch = if (vision.is_some()
            || audio.is_some()
            || audio_chunks.is_some()
            || qwen_vision.is_some()
            || lfm3_vision.is_some())
            && max_batch > 1
        {
            max_batch.max(crate::engine::vision::IMAGE_ROWS)
        } else {
            max_batch
        };
        // On AArch64 the block attention keeps R16 copies of the cache, in whole blocks of 16
        // positions: the reserve is rounded up to them (`--ctx 1000` reserves 1 008).
        let reserved = if cfg!(target_arch = "aarch64") && crate::jit::knobs::knobs().attn_block {
            reserved.next_multiple_of(16)
        } else {
            reserved
        };
        let budget = ModelBudget {
            kv_reserve_tokens: reserved,
            vocab: geo.vocab,
            max_batch,
        };
        // `HERBERT_PREFILL_PROFILE=1`: the prefill regime stamped (the decode one as without), the
        // trace sized for the blocks of a whole reservation in one call (at most 512: a
        // longer call goes by pieces, `prefill`).
        let profile = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
            && !measure.is_on()
            && crate::engine::prefill_profile::requested();
        let profiled;
        let measure = if profile {
            let batch = crate::jit::schedule::prefill_pass()
                .min(max_batch)
                .min(reserved)
                .max(1);
            profiled = Measure {
                sync_trace: true,
                pmu: None,
                trace_blocks: reserved.div_ceil(batch).clamp(1, 512),
                prefill_only: true,
            };
            eprintln!("[engine] HERBERT_PREFILL_PROFILE: prefill regime stamped, trace of {} blocks of {batch} rows", profiled.trace_blocks);
            &profiled
        } else {
            measure
        };
        let widths = geometry::mlp_widths(&model);
        // Keys and values, bf16, on every layer that owns a cache, at the model's KV head count.
        let kv_bytes: u64 = plan_kv
            .iter()
            .map(|p| match p {
                crate::jit::PlanKv::Own(head_dim) => {
                    (4 * geo.shape.n_kv_heads * reserved * head_dim) as u64
                }
                crate::jit::PlanKv::Absent => 0,
            })
            .sum();
        crate::core::memory::refuse_if_it_cannot_fit("KV cache", kv_bytes)?;
        let mut runner = Runner::new_with_geometries(
            geo.shape,
            geo.n_layers,
            budget,
            &tables,
            &widths,
            &geometries,
            &plan_kv,
            geo.delta,
            tuning.executor_options(),
        )?;
        {
            let weights = Self::borrow_weights(&model, &layers);
            runner.mount_execution(
                &weights,
                MountOptions {
                    trace_sync: measure.sync_trace,
                    pmu: measure.pmu,
                    shared_barrier: tuning.shared_barrier,
                    fuse_serial: tuning.fuse_serial,
                    trace_blocks: measure.trace_blocks,
                    trace_prefill_only: measure.prefill_only,
                },
            )?;
        }
        let mount_seconds = mounted.elapsed().as_secs_f64();
        let faulted = Instant::now();
        let touched = Self::fault_in(&model, &layers, threads.max(1));
        eprintln!(
            "[engine] {:.2} GiB of weights touched in {:.3} s",
            touched as f64 / (1u64 << 30) as f64,
            faulted.elapsed().as_secs_f64()
        );
        eprintln!(
            "[engine] loaded in {load_seconds:.1} s, mounted in {mount_seconds:.2} s — {reserved} positions reserved, \
             prefill program {:?}, decode program {:?}",
            runner.execution_code_bytes(true),
            runner.execution_code_bytes(false)
        );
        Ok(Engine {
            layers,
            model,
            delta_packs,
            config,
            geo,
            runner,
            reserved,
            tuning: tuning.clone(),
            threads,
            load_seconds,
            mount_seconds,
            dir: crate::model::loader::resolve_model_dir(model_dir)?,
            vision,
            tower: None,
            image_tokens: Vec::new(),
            audio,
            audio_chunks,
            audio_tower: None,
            qwen_vision,
            qwen_tower: None,
            qwen_table: (0, 0),
            rope_shift: 0,
            rope_clean_from: 0,
            audio_tokens: Vec::new(),
            lfm3_vision,
            lfm3_tower: None,
            lfm3_tokens: Vec::new(),
            prefill_profile: profile.then(Default::default),
        })
    }

    /// The audio tower `(sample rate, max soft tokens, chunk)`, `None` without one this engine
    /// serves (or without the rows of a block: a mixture of experts prefilled one row at a
    /// time). The soft tokens of a recording of `n` samples at that rate:
    /// [`crate::engine::audio::soft_tokens`] for the conformer (`chunk` `None`), one per `chunk`
    /// samples for 12B's encoder-free embedder.
    pub fn audio(&self) -> Option<(usize, usize, Option<usize>)> {
        let chunk = self.audio_chunks.as_ref().map(|c| c.samples);
        if self.audio.is_none() && chunk.is_none() {
            return None;
        }
        if cfg!(target_arch = "riscv64") {
            // Its elementwise kernels are not emitted for RVV (`ElemOp::LocalAttention`,
            // `ElemOp::DepthwiseRows`).
            return None;
        }
        let rows = crate::engine::vision::IMAGE_ROWS;
        (self.runner.max_batch() >= rows && self.reserved >= rows).then_some((
            crate::engine::audio::SAMPLE_RATE,
            crate::engine::audio::soft_tokens(crate::engine::audio::MAX_SAMPLES),
            chunk,
        ))
    }

    /// The recording (`wave`: mono f32 at [`crate::engine::audio::SAMPLE_RATE`], cut at 30 s)
    /// through the audio tower, its `soft` tokens fed at the current position as consecutive
    /// causal positions, in blocks of the image block's rows (an error if the tower makes
    /// another count). The tower and the audio regime are mounted at the first recording.
    /// `HERBERT_JIT_FEATS=FILE` writes the soft tokens in f32, `HERBERT_AUDIO_TIMES=1` the times.
    pub fn prefill_audio(&mut self, wave: &[f32], soft: usize) -> Result<()> {
        if self.audio().is_none() {
            return Err(HerbertError::Backend(
                "no audio block in this engine's reservation".into(),
            ));
        }
        self.room_for(soft)?;
        let rows = crate::engine::vision::IMAGE_ROWS;
        if self.audio_tower.is_none() {
            let t0 = Instant::now();
            let workers = crate::pool::thread_pool::global_pool().num_workers();
            let hidden = self.config.hidden_size;
            let tower = match (self.audio.clone(), self.audio_chunks.clone()) {
                (Some(cfg), _) => {
                    crate::engine::audio::Hearing::Tower(crate::engine::audio::AudioTower::load(
                        &self.dir,
                        "model.",
                        cfg,
                        hidden,
                        &mut self.runner,
                        workers,
                    )?)
                }
                (None, Some(cfg)) => crate::engine::audio::Hearing::Chunks(
                    crate::engine::audio::ChunkEmbedder::load(
                        &self.dir,
                        "model.",
                        cfg,
                        hidden,
                        &self.runner,
                        workers,
                    )?,
                ),
                (None, None) => {
                    return Err(HerbertError::Backend(
                        "this checkpoint has no audio tower this engine serves".into(),
                    ))
                }
            };
            let weights = Self::borrow_weights(&self.model, &self.layers);
            // SAFETY: the table is the tower's own buffer, on the heap; the engine keeps the tower
            // (`self.audio_tower`) for its life.
            unsafe { self.runner.mount_audio(&weights, tower.table(), rows)? };
            self.audio_tokens = (0..tower.max_tokens() as u32).collect();
            self.audio_tower = Some(tower);
            eprintln!(
                "[audio] tower and audio regime ready in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
        }
        let tower = self.audio_tower.as_mut().expect("the tower");
        let (frames, got, prep, tower_s) = tower.encode(&mut self.runner, wave)?;
        if got != soft {
            return Err(HerbertError::Backend(format!(
                "the audio tower made {got} soft tokens, the server counted {soft}"
            )));
        }
        if let Some(path) = std::env::var_os("HERBERT_JIT_FEATS") {
            let n = got * self.config.hidden_size;
            let f: Vec<u8> = tower.table()[..n]
                .iter()
                .flat_map(|&b| f32::from_bits(u32::from(b) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        let t = Instant::now();
        let weights = Self::borrow_weights(&self.model, &self.layers);
        let mut at = 0;
        while at < got {
            let n = (got - at).min(rows);
            self.runner
                .audio_mounted(&weights, &self.audio_tokens[at..at + n])?;
            at += n;
        }
        if std::env::var_os("HERBERT_AUDIO_TIMES").is_some() {
            eprintln!(
                "[audio] {:.2} s → {frames} frames → {soft} soft tokens: preparation {:.1} ms, tower {:.1} ms, fed {:.1} ms ({} threads)",
                wave.len() as f64 / crate::engine::audio::SAMPLE_RATE as f64,
                prep * 1e3,
                tower_s * 1e3,
                t.elapsed().as_secs_f64() * 1e3,
                self.threads
            );
        }
        Ok(())
    }

    /// The vision tower `(patch, pool, max soft tokens)`, `None` without one this engine serves (or
    /// without the rows of an image's block: a mixture of experts prefilled one row at a time).
    pub fn vision(&self) -> Option<(usize, usize, usize)> {
        if let Some(q) = &self.qwen_vision {
            let rows = crate::engine::vision::IMAGE_ROWS;
            return (!cfg!(target_arch = "riscv64")
                && self.runner.max_batch() >= rows
                && self.reserved >= rows)
                .then_some((q.patch_size, q.merge, rows));
        }
        if let Some(v) = &self.lfm3_vision {
            let rows = crate::engine::vision::IMAGE_ROWS;
            if cfg!(target_arch = "riscv64")
                || self.runner.max_batch() < rows
                || self.reserved < rows
            {
                return None;
            }
            return Some((v.patch_size, v.merge, v.max_soft()));
        }
        let v = self.vision.as_ref()?;
        if cfg!(target_arch = "riscv64") {
            // Its elementwise kernels are not emitted for RVV (`ElemOp::Clamp`,
            // `ElemOp::PoolRows`).
            return None;
        }
        let rows = crate::engine::vision::IMAGE_ROWS;
        (self.runner.max_batch() >= rows && self.reserved >= rows).then_some((
            v.patch_size,
            v.pooling_kernel_size,
            rows,
        ))
    }

    /// The image (`rgb`: `h` rows of `w` pixels) through the vision tower, its `soft` tokens fed at
    /// the current position as one block (an error if the tower makes another count). The tower
    /// and the image regime are mounted at the first image. `HERBERT_JIT_FEATS=FILE` writes the
    /// soft tokens in f32, `HERBERT_VISION_TIMES=1` the times.
    pub fn prefill_image(&mut self, rgb: &[u8], h: usize, w: usize, soft: usize) -> Result<()> {
        if self.qwen_vision.is_some() {
            return self.prefill_image_qwen(rgb, h, w, soft);
        }
        if self.lfm3_vision.is_some() {
            return self.prefill_image_lfm3(rgb, h, w, soft);
        }
        let cfg = self.vision.clone().ok_or_else(|| {
            HerbertError::Backend("this checkpoint has no vision tower this engine serves".into())
        })?;
        if self.vision().is_none() {
            return Err(HerbertError::Backend(
                "no image block in this engine's reservation".into(),
            ));
        }
        self.room_for(soft)?;
        if self.tower.is_none() {
            let t0 = Instant::now();
            let workers = crate::pool::thread_pool::global_pool().num_workers();
            let tower = crate::engine::vision::Encoder::load(
                &self.dir,
                "model.",
                cfg,
                self.config.hidden_size,
                &self.runner,
                workers,
            )?;
            let rows = crate::engine::vision::IMAGE_ROWS;
            let weights = Self::borrow_weights(&self.model, &self.layers);
            // SAFETY: the table is the tower's own buffer, on the heap; the engine keeps the tower
            // (`self.tower`) for its life.
            unsafe { self.runner.mount_image(&weights, tower.table(), rows)? };
            self.tower = Some(tower);
            self.image_tokens = (0..rows as u32).collect();
            eprintln!(
                "[vision] tower and image regime ready in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
        }
        let tower = self.tower.as_mut().expect("the tower");
        let (gw, gh, got, prep, tower_s) = tower.encode(&mut self.runner, rgb, h, w)?;
        if got != soft {
            return Err(HerbertError::Backend(format!(
                "the tower made {got} soft tokens, the server counted {soft}"
            )));
        }
        if let Some(path) = std::env::var_os("HERBERT_JIT_FEATS") {
            let n = got * self.config.hidden_size;
            let f: Vec<u8> = tower.table()[..n]
                .iter()
                .flat_map(|&b| f32::from_bits(u32::from(b) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        let t = Instant::now();
        let weights = Self::borrow_weights(&self.model, &self.layers);
        self.runner
            .image_mounted(&weights, &self.image_tokens[..soft])?;
        if std::env::var_os("HERBERT_VISION_TIMES").is_some() {
            eprintln!(
                "[vision] {w}×{h} → {gw}×{gh} patches → {soft} soft tokens: preparation {:.1} ms, tower {:.1} ms, fed {:.1} ms ({} threads)",
                prep * 1e3,
                tower_s * 1e3,
                t.elapsed().as_secs_f64() * 1e3,
                self.threads
            );
        }
        Ok(())
    }

    /// Qwen: the soft tokens of an `h × w` image and the RoPE positions they take (`None`: no Qwen
    /// tower).
    pub fn image_tokens(&self, h: usize, w: usize) -> Option<Result<(usize, usize)>> {
        self.qwen_vision.as_ref().map(|q| q.tokens(h, w))
    }

    /// Qwen's multimodal RoPE: the text fed from the current position on turns by its K/V position
    /// minus `shift` (the positions the images before it saved, `Σ soft − max(grid_h, grid_w)`,
    /// which the server counts from the prompt). The RoPE rows from the current position on are
    /// written again when they hold another offset or an image's rows. Without a Qwen tower:
    /// nothing (the tables are never touched).
    pub fn set_rope_shift(&mut self, shift: usize) -> Result<()> {
        if self.qwen_vision.is_none() {
            return Ok(());
        }
        let pos = self.runner.pos();
        if shift > pos {
            return Err(HerbertError::Backend(format!(
                "a RoPE shift of {shift} at position {pos}"
            )));
        }
        if shift == self.rope_shift && pos >= self.rope_clean_from {
            return Ok(());
        }
        self.write_text_rope(pos, shift)?;
        Ok(())
    }

    /// The RoPE rows of positions `from..` for text at `shift`: row `p` the rotation of position `p
    /// − shift`.
    fn write_text_rope(&mut self, from: usize, shift: usize) -> Result<()> {
        let q = self.qwen_vision.clone().expect("a Qwen tower");
        let inv = crate::model::loader::rope_inv_freq(&self.config);
        let rn = self.config.rotary_ndims;
        let half = rn / 2;
        let (cos, sin) = self
            .runner
            .rope_rows_mut(rn)
            .ok_or_else(|| HerbertError::Backend(format!("no RoPE table of {rn}")))?;
        let rows = cos.len() / half;
        for p in from..rows {
            let t = p - shift;
            crate::engine::vision_qwen::mrope_bf16(
                &q,
                &inv,
                [t; 3],
                &mut cos[p * half..(p + 1) * half],
                &mut sin[p * half..(p + 1) * half],
            );
        }
        self.rope_shift = shift;
        self.rope_clean_from = from;
        Ok(())
    }

    /// Qwen3.5 / Qwen3.6 / Qwen3-VL: the image through the tower (read at the first image, lowered
    /// again for a larger one), its `soft` tokens fed at the current position as causal
    /// positions by blocks of the image block's rows, each turning by its `(t, h, w)`
    /// (`get_rope_index`: `t` the text position `p − shift`, `h` and `w` that plus the soft token's
    /// row and column in the merged grid); the text after it shifted by `soft − max(rows,
    /// columns)` more. Qwen3-VL: before each block, the deepstack vectors of its rows into the
    /// language model's slots (added after its first layers). `HERBERT_VISION_DEEPSTACK=FILE`
    /// reads them from a reference, `HERBERT_JIT_DEEPSTACK=FILE` writes the tower's
    /// (f32, `[deepstack][soft][hidden]`).
    #[allow(clippy::manual_checked_ops)] // `stride` only under `lm_deep > 0`
    fn prefill_image_qwen(&mut self, rgb: &[u8], h: usize, w: usize, soft: usize) -> Result<()> {
        let cfg = self.qwen_vision.clone().expect("a Qwen tower");
        if self.vision().is_none() {
            return Err(HerbertError::Backend(
                "no image block in this engine's reservation".into(),
            ));
        }
        self.room_for(soft)?;
        let (want, _) = cfg.tokens(h, w)?;
        if want != soft {
            return Err(HerbertError::Backend(format!(
                "the image makes {want} soft tokens, the server counted {soft}"
            )));
        }
        if self.qwen_tower.is_none() {
            let workers = crate::pool::thread_pool::global_pool().num_workers();
            self.qwen_tower = Some(crate::engine::vision_qwen::QwenTower::load(
                &self.dir,
                cfg.clone(),
                &self.runner,
                workers,
            )?);
        }
        let tower = self.qwen_tower.as_mut().expect("the tower");
        let (mh, mw, _, prep, tower_s) = tower.encode(&mut self.runner, rgb, h, w)?;
        // `HERBERT_VISION_SOFT=FILE`: the soft tokens read from `FILE` (f32, `[soft][hidden]`, a
        // reference's merger output) in place of the tower's, to compare the language model
        // alone.
        if let Some(path) = std::env::var_os("HERBERT_VISION_SOFT") {
            let bytes = std::fs::read(path)?;
            let n = soft * self.config.hidden_size;
            if bytes.len() != n * 4 {
                return Err(HerbertError::Backend(format!(
                    "HERBERT_VISION_SOFT: {} bytes, {} expected",
                    bytes.len(),
                    n * 4
                )));
            }
            for (d, c) in tower.table_mut()[..n]
                .iter_mut()
                .zip(bytes.as_chunks::<4>().0)
            {
                *d = crate::engine::packs::to_bf16(f32::from_le_bytes(*c));
            }
        }
        // Qwen3-VL, `HERBERT_VISION_DEEPSTACK=FILE`: the deepstack vectors read from `FILE` (f32,
        // `[deepstack][soft][hidden]`, a reference's `deepstack_features`), with
        // `HERBERT_VISION_SOFT`.
        let n_deep = cfg.deepstack.iter().filter(|&&d| d < cfg.depth).count();
        if let Some(path) = std::env::var_os("HERBERT_VISION_DEEPSTACK") {
            let bytes = std::fs::read(path)?;
            let hidden = self.config.hidden_size;
            let max_soft = tower.deepstack().1;
            if bytes.len() != n_deep * soft * hidden * 4 {
                return Err(HerbertError::Backend(format!(
                    "HERBERT_VISION_DEEPSTACK: {} bytes, {} expected",
                    bytes.len(),
                    n_deep * soft * hidden * 4
                )));
            }
            let deep = tower.deepstack_mut();
            for i in 0..n_deep {
                for (j, c) in bytes[i * soft * hidden * 4..(i + 1) * soft * hidden * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .enumerate()
                {
                    deep[i * max_soft * hidden + j] =
                        crate::engine::packs::to_bf16(f32::from_le_bytes(*c));
                }
            }
        }
        let table = tower.table();
        let (deep, max_soft) = tower.deepstack();
        // The language model adds `lm_deep` (a tower cut short by `HERBERT_VISION_DEPTH` makes
        // fewer: zeros after).
        let lm_deep = self
            .config
            .qwen_vl
            .as_ref()
            .map_or(0, |v| v.deepstack_layers);
        if n_deep > lm_deep || (n_deep == 0) != (lm_deep == 0) {
            return Err(HerbertError::Backend(format!(
                "the tower makes {n_deep} deepstack vectors, the language model adds {lm_deep}"
            )));
        }
        if let Some(path) = std::env::var_os("HERBERT_JIT_DEEPSTACK") {
            let hidden = self.config.hidden_size;
            let f: Vec<u8> = (0..n_deep)
                .flat_map(|i| deep[i * max_soft * hidden..(i * max_soft + soft) * hidden].iter())
                .flat_map(|&b| f32::from_bits(u32::from(b) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        if mh * mw != soft {
            return Err(HerbertError::Backend(format!(
                "the tower made {} soft tokens, the server counted {soft}",
                mh * mw
            )));
        }
        if let Some(path) = std::env::var_os("HERBERT_JIT_FEATS") {
            let n = soft * self.config.hidden_size;
            let f: Vec<u8> = table[..n]
                .iter()
                .flat_map(|&b| f32::from_bits(u32::from(b) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        let rows = crate::engine::vision::IMAGE_ROWS;
        let weights = Self::borrow_weights(&self.model, &self.layers);
        if self.qwen_table != (table.as_ptr() as usize, table.len()) {
            let t0 = Instant::now();
            // SAFETY: the table of the tower the engine keeps; a table at another address is
            // mounted again (`qwen_table`).
            unsafe { self.runner.mount_causal_media(&weights, table, rows)? };
            self.qwen_table = (table.as_ptr() as usize, table.len());
            self.audio_tokens = (0..(table.len() / self.config.hidden_size) as u32).collect();
            eprintln!(
                "[vision] image regime mounted on {} soft tokens in {:.2} s",
                self.audio_tokens.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        // The image's RoPE rows at its K/V positions.
        let pos = self.runner.pos();
        let start = pos - self.rope_shift;
        let inv = crate::model::loader::rope_inv_freq(&self.config);
        let half = self.config.rotary_ndims / 2;
        {
            let (cos, sin) = self
                .runner
                .rope_rows_mut(self.config.rotary_ndims)
                .ok_or_else(|| HerbertError::Backend("no RoPE table".into()))?;
            for r in 0..soft {
                let p = pos + r;
                let (i, j) = (r / mw, r % mw);
                crate::engine::vision_qwen::mrope_bf16(
                    &cfg,
                    &inv,
                    [start, start + i, start + j],
                    &mut cos[p * half..(p + 1) * half],
                    &mut sin[p * half..(p + 1) * half],
                );
            }
        }
        let t = Instant::now();
        let mut at = 0;
        let hidden = self.config.hidden_size;
        while at < soft {
            let n = (soft - at).min(rows);
            // Qwen3-VL: the deepstack vectors of the block's rows, row `r` of layer `i` for row `r`
            // of the block.
            if lm_deep > 0 {
                let block = self.runner.deepstack_mut();
                let stride = block.len() / lm_deep;
                for i in 0..lm_deep {
                    let dst = &mut block[i * stride..i * stride + n * hidden];
                    if i < n_deep {
                        dst.copy_from_slice(
                            &deep[(i * max_soft + at) * hidden..(i * max_soft + at + n) * hidden],
                        );
                    } else {
                        dst.fill(0);
                    }
                }
            }
            self.runner
                .audio_mounted(&weights, &self.audio_tokens[at..at + n])?;
            at += n;
        }
        let fed = t.elapsed().as_secs_f64();
        // The text after the image: `max(rows, columns)` positions past its start.
        let shift = self.rope_shift + soft - mh.max(mw);
        self.write_text_rope(pos + soft, shift)?;
        if std::env::var_os("HERBERT_VISION_TIMES").is_some() {
            eprintln!(
                "[vision] {w}×{h} → {}×{} patches → {soft} soft tokens: preparation {:.1} ms, tower {:.1} ms, fed {:.1} ms ({} threads)",
                mw * cfg.merge,
                mh * cfg.merge,
                prep * 1e3,
                tower_s * 1e3,
                fed * 1e3,
                self.threads
            );
        }
        Ok(())
    }

    /// LFM3-VL: the soft tokens of an `h × w` image (`smart_resize`), `None` for another model.
    pub fn image_soft_tokens(
        &self,
        h: usize,
        w: usize,
    ) -> Option<std::result::Result<usize, String>> {
        self.lfm3_vision.as_ref().map(|v| v.soft_tokens(h, w))
    }

    /// LFM3-VL's image: through its tower and projector, the soft tokens fed at the current
    /// position as consecutive causal positions, in blocks of the media regime's rows (the
    /// regime of Gemma's recordings: the rows read from the tower's table).
    fn prefill_image_lfm3(&mut self, rgb: &[u8], h: usize, w: usize, soft: usize) -> Result<()> {
        if self.vision().is_none() {
            return Err(HerbertError::Backend(
                "no image block in this engine's reservation".into(),
            ));
        }
        self.room_for(soft)?;
        let rows = crate::engine::vision::IMAGE_ROWS;
        if self.lfm3_tower.is_none() {
            let t0 = Instant::now();
            let cfg = self.lfm3_vision.clone().expect("LFM3-VL");
            let workers = crate::pool::thread_pool::global_pool().num_workers();
            let tower = crate::engine::vision_lfm3::Lfm3Tower::load(
                &self.dir,
                "model.",
                cfg,
                self.config.hidden_size,
                &self.runner,
                workers,
            )?;
            let weights = Self::borrow_weights(&self.model, &self.layers);
            // SAFETY: the table is the tower's own buffer, on the heap; the engine keeps the tower
            // (`self.lfm3_tower`) for its life.
            unsafe { self.runner.mount_audio(&weights, tower.table(), rows)? };
            self.lfm3_tokens = (0..tower.max_tokens() as u32).collect();
            self.lfm3_tower = Some(tower);
            eprintln!(
                "[vision] LFM3 tower and media regime ready in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
        }
        let tower = self.lfm3_tower.as_mut().expect("the tower");
        let (gw, gh, got, prep, tower_s) = tower.encode(&mut self.runner, rgb, h, w)?;
        if got != soft {
            return Err(HerbertError::Backend(format!(
                "the tower made {got} soft tokens, the server counted {soft}"
            )));
        }
        if let Some(path) = std::env::var_os("HERBERT_JIT_FEATS") {
            let n = got * self.config.hidden_size;
            let f: Vec<u8> = tower.table()[..n]
                .iter()
                .flat_map(|&b| f32::from_bits(u32::from(b) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        let t = Instant::now();
        let weights = Self::borrow_weights(&self.model, &self.layers);
        let mut at = 0;
        while at < got {
            let n = (got - at).min(rows);
            self.runner
                .audio_mounted(&weights, &self.lfm3_tokens[at..at + n])?;
            at += n;
        }
        if std::env::var_os("HERBERT_VISION_TIMES").is_some() {
            eprintln!(
                "[vision] {w}×{h} → {gw}×{gh} patches → {soft} soft tokens: preparation {:.1} ms, tower {:.1} ms, fed {:.1} ms ({} threads)",
                prep * 1e3,
                tower_s * 1e3,
                t.elapsed().as_secs_f64() * 1e3,
                self.threads
            );
        }
        Ok(())
    }

    /// Reads one word per 4 KiB of every matrix, `threads` at a time: a page
    /// the system compressed or never mapped is brought back here rather than
    /// during the first prompt. On the M5, with the swap nearly full, macOS
    /// compresses the packed weights as soon as they are written; the first
    /// prefill then spent 0.9 s decompressing them one fault at a time (1.1 s
    /// for 20 tokens, 0.19 s once touched here, in 0.4 s). Returns the bytes
    /// covered.
    fn fault_in(model: &Model, layers: &[LayerWeights<'_>], threads: usize) -> usize {
        Self::touch(&Self::matrices(model, layers), threads)
    }

    /// Every matrix the engine reads: the head, the projections, the experts.
    fn matrices<'a>(model: &'a Model, layers: &'a [LayerWeights<'_>]) -> Vec<&'a [u16]> {
        let mut slices: Vec<&[u16]> = vec![&model.lm_head.data_r16];
        for l in layers {
            slices.extend([l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down]);
            if let crate::jit::Mixer::GatedDeltaNet(d) = &l.mixer {
                slices.extend([
                    d.in_proj_qkv,
                    d.in_proj_z,
                    d.in_proj_a,
                    d.in_proj_b,
                    d.out_proj,
                ]);
            }
            if let Some(m) = &l.qwen_moe {
                slices.extend(m.router.iter().chain(&m.gate_up).chain(&m.down).copied());
            }
        }
        slices
    }

    /// One word per 4 KiB of `slices` read, `threads` at a time: the bytes
    /// covered, 0 when the pool could not run it.
    fn touch(slices: &[&[u16]], threads: usize) -> usize {
        let bytes = slices.iter().map(|s| s.len() * 2).sum();
        // The workers of the pool, configured just before, each taking the
        // next matrix: no threads of its own.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let pool = crate::pool::thread_pool::global_pool();
        let touch = |_: usize| loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(slice) = slices.get(i) else { break };
            let mut sum = 0u16;
            for j in (0..slice.len()).step_by(2048) {
                sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(&slice[j]) });
            }
            std::hint::black_box(sum);
        };
        if pool
            .collective(threads.min(pool.num_workers()).max(1), touch, || {})
            .is_err()
        {
            // A pool that cannot run a collective leaves the pages to the first prompt.
            return 0;
        }
        bytes
    }

    fn borrow_weights<'a>(model: &'a Model, layers: &'a [LayerWeights<'a>]) -> ModelWeights<'a> {
        ModelWeights {
            embed: &model.embed_tokens,
            layers,
            final_norm: &model.norm,
            lm_head: &model.lm_head.data_r16,
            gemma: model.gemma.as_ref().map(|g| crate::jit::GemmaModelWeights {
                ple: g.ple.as_ref().map(|p| crate::jit::PleWeights {
                    table: &p.table,
                    model_proj: &p.model_proj.data_r16,
                    proj_norm: &p.proj_norm,
                }),
            }),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geo
    }

    pub fn runner(&self) -> &Runner {
        &self.runner
    }

    /// Positions already in the state: prompt rows and fed tokens.
    pub fn pos(&self) -> usize {
        self.runner.pos()
    }

    /// The ceiling of a conversation, in positions.
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Back to position zero: KV cache, DeltaNet states and position cleared.
    pub fn reset(&mut self) {
        self.runner.reset();
        if self.qwen_vision.is_some() && (self.rope_shift != 0 || self.rope_clean_from != 0) {
            let _ = self.write_text_rope(0, 0);
        }
    }

    /// Keep the first `p` positions of the state (`p ≤ pos`): what the server
    /// keeps of the previous request when the next one shares that prefix.
    pub fn truncate(&mut self, p: usize) -> Result<()> {
        self.runner.truncate(p)
    }

    /// Append `tokens` to the live state from the current position, and return
    /// the greedy token of the last row. The second turn of a conversation calls
    /// this with only its new rows.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        if tokens.is_empty() {
            return Err(HerbertError::Backend("prefill of zero tokens".to_string()));
        }
        self.room_for(tokens.len())?;
        if self.prefill_profile.is_some()
            && tokens.len().div_ceil(self.prefill_block()) > self.profile_blocks()
        {
            // More blocks than the trace holds: by pieces, as `prefill_until`.
            return self
                .prefill_until(tokens, &mut || false)
                .map(|t| t.expect("never stopped"));
        }
        let weights = Self::borrow_weights(&self.model, &self.layers);
        let sw = crate::engine::prefill_profile::context_switches();
        let t = std::time::Instant::now();
        self.runner.prefill_mounted(&weights, tokens, None)?;
        if self.prefill_profile.is_some() {
            self.profile_fold(tokens.len(), t.elapsed().as_secs_f64(), sw);
            if let Some(p) = self.prefill_profile.as_mut() {
                p.report();
            }
        }
        Ok(self.argmax())
    }

    /// The blocks the profile's trace holds per call.
    fn profile_blocks(&self) -> usize {
        let block = self.prefill_block();
        self.reserved.div_ceil(block).clamp(1, 512)
    }

    /// The trace of the prefill call that just ran into the profile.
    fn profile_fold(&mut self, rows: usize, seconds: f64, switches_before: (i64, i64)) {
        let after = crate::engine::prefill_profile::context_switches();
        let switches = (after.0 - switches_before.0, after.1 - switches_before.1);
        let (Some(plan), Some(trace)) = (
            self.runner.execution_plan(true),
            self.runner.execution_sync_trace(),
        ) else {
            return;
        };
        if let Some(p) = self.prefill_profile.as_mut() {
            p.fold(plan.labels(), &trace, rows, seconds, switches);
        }
    }

    /// Decode in one launch of the generated program: `first` is fed to block 0,
    /// then `step(argmax, logits_bf16)` is called at the end of every block and
    /// returns the token to feed next, or `None` to stop. Returns the number of
    /// blocks run.
    pub fn generate(
        &mut self,
        first: u32,
        max_steps: usize,
        step: &mut dyn FnMut(u32, &[u16]) -> Result<Option<u32>>,
    ) -> Result<usize> {
        let room = self.reserved.saturating_sub(self.runner.pos());
        let max_steps = max_steps.min(room);
        if max_steps == 0 {
            return Ok(0);
        }
        let weights = Self::borrow_weights(&self.model, &self.layers);
        self.runner
            .generate_mounted(&weights, first, max_steps, step)
    }

    /// The last logits row, bf16 bits, `vocab_size` long.
    pub fn logits(&self) -> &[u16] {
        &self.runner.logits()[..self.config.vocab_size]
    }

    /// The hidden state of the last token fed after the final norm, bf16 bits, `hidden_size` long.
    pub fn final_hidden(&mut self) -> Vec<u16> {
        self.runner.final_hidden()
    }

    /// The greedy token of the last logits row.
    pub fn argmax(&self) -> u32 {
        self.runner
            .execution_argmax()
            .unwrap_or_else(|| argmax_bf16(self.logits()))
    }

    /// The whole state at the current position, frozen.
    pub fn snapshot(&self) -> Result<Instantaneous> {
        let mut slot = self.runner.snapshot_reserve(self.reserved);
        self.runner.snapshot_take(&mut slot)?;
        Ok(slot)
    }

    /// An empty snapshot sized for the whole reservation: the only allocation of a saved state.
    pub fn snapshot_reserve(&self) -> Instantaneous {
        self.runner.snapshot_reserve(self.reserved)
    }

    /// The whole state at the current position into `slot`, without allocating.
    pub fn snapshot_into(&self, slot: &mut Instantaneous) -> Result<()> {
        self.runner.snapshot_take(slot)
    }

    /// Replace the state by `slot`, rewound to position `p`.
    pub fn restore(&mut self, slot: &Instantaneous, p: usize) -> Result<()> {
        self.runner.reset();
        self.runner.restore_from(slot, p)
    }

    /// [`Self::prefill`], `stop` asked before each piece of the prompt: once it
    /// says so, `None`, the state then holding what was fed until there. The
    /// pieces are whole multiples of the prefill's block from the first token,
    /// so that the blocks fall where one call puts them: the same bytes. Only
    /// the last piece runs the final norm and the language head (Ryzen 9 9900X,
    /// Qwen3.5-2B `--q4`, 4 682 tokens by pieces of 256 rows: 842-867 tok/s
    /// with the head on every piece, 878-903 in one call; without it 891-947
    /// against 900-926).
    pub fn prefill_until(
        &mut self,
        tokens: &[u32],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<u32>> {
        if tokens.is_empty() {
            return Err(HerbertError::Backend("prefill of zero tokens".to_string()));
        }
        let rows = self.stop_rows();
        if let Some(p) = self.prefill_profile.as_mut() {
            // A prefill stopped before its end left its pieces here.
            if !p.is_empty() {
                **p = Default::default();
            }
        }
        let last = (tokens.len() - 1) / rows;
        for (i, piece) in tokens.chunks(rows).enumerate() {
            if stop() {
                return Ok(None);
            }
            if i == last {
                return self.prefill(piece).map(Some);
            }
            self.room_for(piece.len())?;
            let weights = Self::borrow_weights(&self.model, &self.layers);
            let sw = crate::engine::prefill_profile::context_switches();
            let t = std::time::Instant::now();
            self.runner.prefill_mounted_headless(&weights, piece)?;
            if self.prefill_profile.is_some() {
                self.profile_fold(piece.len(), t.elapsed().as_secs_f64(), sw);
            }
        }
        unreachable!("the last piece returns")
    }

    /// An error when `rows` more would pass the reserved positions.
    fn room_for(&self, rows: usize) -> Result<()> {
        if self.runner.pos() + rows > self.reserved {
            return Err(HerbertError::Backend(format!(
                "the conversation exceeds the {} reserved positions ({} in the state, {} to add)",
                self.reserved,
                self.runner.pos(),
                rows
            )));
        }
        Ok(())
    }

    /// The rows between two questions of [`Self::prefill_until`]: the blocks
    /// of one prefill pass (`--prefill-pass`), at least one block.
    pub fn stop_rows(&self) -> usize {
        let block = self.runner.prefill_batch().unwrap_or(1).max(1);
        block * (causal_batch_width() / block).max(1)
    }

    /// The positions a reused prefix is cut to a multiple of ([`Runner::block_align`]).
    pub fn block_align(&self) -> usize {
        self.runner.block_align()
    }

    /// Rows of one block of the mounted prefill.
    pub fn prefill_block(&self) -> usize {
        self.runner.prefill_batch().unwrap_or(1)
    }

    /// What the K/V of a prefix depends on besides its tokens: the
    /// executable, the machine's instructions, the code of both generated
    /// programs (the kernels, the workers they were lowered for), the tuning,
    /// the workers, the capacity, the configuration. `None` on a model with
    /// DeltaNet layers (the K/V is not its whole state) or when the
    /// executable cannot be read.
    pub fn kv_fingerprint(&self) -> Option<String> {
        if self.runner.has_recurrence() {
            return None;
        }
        static EXE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let exe = EXE
            .get_or_init(|| {
                let bytes = std::fs::read(std::env::current_exe().ok()?).ok()?;
                Some(format!("{:016x} ({} bytes)", fnv64(&bytes), bytes.len()))
            })
            .as_deref()?;
        let code = |prefill| {
            self.runner
                .execution_code_digest(prefill)
                .map(|(h, n)| format!("{h:016x} ({n} bytes)"))
        };
        Some(format!(
            "executable {exe}; isa {}; prefill program {:?}; decode program {:?}; threads {}; capacity {}; tuning {:?}; config {:?}",
            crate::core::isa::summary(),
            code(true),
            code(false),
            self.threads,
            self.reserved,
            self.tuning,
            self.config,
        ))
    }

    /// The K/V of positions `0..n` as bytes.
    pub fn kv_save(&self, n: usize) -> Result<Vec<u8>> {
        self.runner.kv_save(n)
    }

    /// The bytes of [`Self::kv_save`]`(n)` read back: the state is then at `n`.
    pub fn kv_load(&mut self, n: usize, bytes: &[u8]) -> Result<()> {
        self.runner.reset();
        self.runner.kv_load(n, bytes)
    }

    /// An empty [`Recurrence`]: the only allocation of one.
    pub fn recurrence_reserve(&self) -> Recurrence {
        self.runner.recurrence_reserve()
    }

    /// The DeltaNet states at the current position into `r`, the K/V left in place.
    pub fn recurrence_take(&self, r: &mut Recurrence) -> Result<()> {
        self.runner.recurrence_take(r)
    }

    /// `r` back, with the K/V as it is: the position it stands for.
    pub fn recurrence_restore(&mut self, r: &Recurrence) -> Result<usize> {
        self.runner.recurrence_restore(r)
    }

    /// An empty [`KvLane`] (the K/V of the whole reservation).
    pub fn lane_reserve(&self) -> KvLane {
        self.runner.lane_reserve()
    }

    /// The live K/V (`live` positions) and `lane` (`held` positions) exchanged: the bytes moved
    /// each way.
    pub fn lane_swap(&mut self, lane: &mut KvLane, live: usize, held: usize) -> Result<usize> {
        self.runner.lane_swap(lane, live, held)
    }

    /// One word per 4 KiB of every matrix read again (the load's `fault_in`):
    /// the bytes covered.
    pub fn touch_weights(&self) -> usize {
        Self::touch(
            &Self::matrices(&self.model, &self.layers),
            self.threads.max(1),
        )
    }

    /// The trace of the latest call of one regime, when it was mounted with
    /// measurements; nothing otherwise.
    pub fn write_trace(&self, prefill: bool, path: &Path) -> Result<()> {
        crate::engine::trace::write_execution_trace(&self.runner, prefill, path)
            .map_err(|e| HerbertError::Backend(format!("trace {}: {e}", path.display())))
    }
}

/// FNV-1a 64.
fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Argmax on bf16 logits, comparing the reconstructed f32 values.
pub fn argmax_bf16(logits: &[u16]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &b) in logits.iter().enumerate() {
        let v = f32::from_bits((b as u32) << 16);
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}
