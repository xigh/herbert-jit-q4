//! What `main` decides about the engine, as named fields.
//!
//! Every setting `main` passes is a field of one of these two structs, built in `main` with its
//! default here, and a sentence that says what it changes. Apart from them, a few `HERBERT_*`
//! environment variables, read where they apply, turn on profiles and timings, override a limit
//! for an experiment, or mask CPU features (`HERBERT_NO_AVX512`, `HERBERT_NO_BF16`,
//! `core/isa.rs`). `Engine::load` installs `Tuning`
//! once per process before the pool is created and before any kernel is
//! compiled, so the emitters and the pool read the same values.
//!
//! `Tuning` is about speed and costs nothing to read; `Measure` is about
//! looking inside a call and costs a little (stamps and counters per node).

use crate::core::error::{HerbertError, Result};
use crate::jit::execution::{Executor, ExecutorOptions, PmuEvent};
pub use crate::jit::knobs::Knobs;
pub use crate::pool::settings::PoolSettings;

/// The knobs of the generated program, of the emitters and of the pool.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    /// The executor of each regime: spin barriers (every participant busy-waits
    /// at a barrier) or hybrid (spin, then sleep after `hybrid_spin` pauses).
    pub prefill_executor: Executor,
    pub decode_executor: Executor,
    /// Participants of the prefill; `None` = every worker of the pool.
    pub prefill_workers: Option<usize>,
    /// Participants of the decode; `None` = every worker of the pool. By
    /// default sixteen on a Mac with more than sixteen performance cores,
    /// the whole pool elsewhere (`cpu_detection::default_decode_workers`:
    /// M3 Ultra, pool of 20, 27.37 → 27.69 tokens/s).
    pub decode_workers: Option<usize>,
    /// Pauses before a hybrid participant sleeps.
    pub hybrid_spin: u32,
    /// One barrier word shared by every node instead of one per node.
    pub shared_barrier: bool,
    /// Fuse the barriers around serial nodes (no gain measured).
    pub fuse_serial: bool,
    /// The emitters and the lowering: pass width, tile, SDPA, DeltaNet, teams.
    pub kernels: Knobs,
    /// The worker pool: spin, pinning, who computes.
    pub pool: PoolSettings,
    /// A mixture-of-experts model prefills one row at a time, as it decodes (the rows of a block
    /// are grouped by expert otherwise, on x86-64).
    pub moe_row_prefill: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            prefill_executor: crate::core::cpu_detection::default_prefill_executor(),
            decode_executor: Executor::SpmdSpin,
            prefill_workers: None,
            decode_workers: crate::core::cpu_detection::default_decode_workers(),
            hybrid_spin: crate::core::cpu_detection::default_hybrid_spin(),
            shared_barrier: false,
            fuse_serial: false,
            kernels: Knobs::default(),
            pool: PoolSettings::default(),
            moe_row_prefill: false,
        }
    }
}

impl Tuning {
    pub fn executor_options(&self) -> ExecutorOptions {
        ExecutorOptions {
            prefill: self.prefill_executor,
            decode: self.decode_executor,
            prefill_workers: self.prefill_workers,
            decode_workers: self.decode_workers,
            hybrid_spin: self.hybrid_spin,
        }
    }

    /// Installs the kernel knobs and the pool settings for the process.
    pub fn install(&self) -> Result<()> {
        let kernels = self.kernels;
        // `--int8` / `--q4` on x86-64: `gemm_int8.rs` on AVX-512 VNNI (`zmm`), AVX-VNNI (`ymm`), or
        // AVX2 + FMA without VNNI (Zen 2, Zen 3, Intel Haswell to Comet Lake: the `ymm`
        // kernel with `vpdpbusd` emulated, the bits of AVX-VNNI in Q4), which the engine's floor
        // (AVX2 + FMA) guarantees; the SSE4.2 micro-kernel only when `--q4-sse42` /
        // `--int8-sse42` force it.
        #[cfg(target_arch = "x86_64")]
        if kernels.neon.w_int8
            && !kernels.neon.w_sse42
            && !crate::core::isa::feature("avxvnni")
            && crate::jit::arch::x86::gemm_int8::available()
        {
            eprintln!(
                "[int8] no AVX-VNNI: the {} projections on the {} micro-kernel",
                if kernels.neon.w_q4 { "Q4" } else { "int8" },
                crate::jit::arch::x86::gemm_int8::name(kernels.neon.w_q4)
            );
        }
        // AArch64: the int8 and Q4 kernels are `sdot` (FEAT_DotProd), the f16 ones `fmla .8h`
        // (FEAT_FP16); a core without them (Cortex-A72, v8.0) refuses the flag instead of
        // stopping on an illegal instruction.
        #[cfg(target_arch = "aarch64")]
        for (asked, feature, flag) in [
            (kernels.neon.w_int8, "dotprod", "--int8 / --q4"),
            (kernels.neon.w_f16, "fp16", "--f16"),
        ] {
            if asked && !crate::core::isa::feature(feature) {
                return Err(HerbertError::Backend(format!(
                    "{flag}: this CPU has no {feature}"
                )));
            }
        }
        // RISC-V: no int8 or Q4 kernel yet; the weights would stay bf16 while the memory check
        // counted them at a third.
        #[cfg(target_arch = "riscv64")]
        if kernels.neon.w_int8 {
            return Err(HerbertError::Backend(
                "--int8 / --q4: not on RISC-V yet (the weights stay bf16 without them)".into(),
            ));
        }
        crate::jit::knobs::install(kernels).map_err(HerbertError::Backend)?;
        crate::pool::settings::install(self.pool.clone()).map_err(HerbertError::Backend)
    }
}

/// What to record inside a call.
#[derive(Clone, Copy, Debug, Default)]
pub struct Measure {
    /// `rdtscp` stamps per node and participant: work, progress, barrier, sleep.
    pub sync_trace: bool,
    /// One hardware counter read by `rdpmc` around every node (Linux, user RDPMC).
    pub pmu: Option<PmuEvent>,
    /// Blocks kept in the trace buffers: a prefill pass is one block, a
    /// decoded token is one block.
    pub trace_blocks: usize,
    /// The prefill regime only (the decode program emitted without stamps).
    pub prefill_only: bool,
}

impl Measure {
    pub fn none() -> Self {
        Measure::default()
    }

    pub fn is_on(&self) -> bool {
        self.sync_trace || self.pmu.is_some()
    }

    /// Time stamps only.
    pub fn sync(trace_blocks: usize) -> Self {
        Measure {
            sync_trace: true,
            pmu: None,
            trace_blocks,
            prefill_only: false,
        }
    }

    /// One PMU event, with the stamps.
    pub fn pmu(event: PmuEvent, trace_blocks: usize) -> Self {
        Measure {
            sync_trace: true,
            pmu: Some(event),
            trace_blocks,
            prefill_only: false,
        }
    }
}
