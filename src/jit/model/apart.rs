//! A program lowered apart from the regimes of a model (the vision tower's, an
//! encoder's) and the executor that runs it: the runner of a chat model, or a
//! [`Standalone`] executor of its own for an engine that has no chat model
//! (EmbeddingGemma 2, XLM-RoBERTa: an encoder and its towers, nothing else).

use crate::core::error::{HerbertError, Result};
use crate::jit::execution::{
    build::Mounted, ExecutionSession, Executor, ExecutorOptions, SyncTraceSnapshot,
};
use crate::pool::thread_pool::global_pool;

fn error(e: impl std::fmt::Display) -> HerbertError {
    HerbertError::Backend(e.to_string())
}

/// What runs a program lowered apart: its team, its session, its trace.
pub(crate) trait Apart {
    /// The spin of the hybrid executor (`None`: spin only), for the program's barriers.
    fn apart_spin(&self) -> Option<u32>;
    /// Run `program` over `tokens.len()` rows in one block from position 0. `tokens` holds the
    /// words the frame points at (no node reads them).
    fn run_apart(&mut self, program: &mut Mounted, tokens: &[u32]) -> Result<()>;
    /// Room for the synchronization trace of `program` run over one block (`rdtscp` stamps).
    fn reserve_apart_trace(&mut self, program: &Mounted) -> Result<()>;
    /// The stamps of the last traced call.
    fn apart_sync_trace(&self) -> Option<SyncTraceSnapshot>;
}

/// The executor of an engine without a chat model: the prefill's options of the tuning, a session
/// of its own.
pub struct Standalone {
    options: ExecutorOptions,
    session: ExecutionSession,
}

impl Standalone {
    /// An executor with `options` (the tuning's), on the global pool.
    pub fn new(options: ExecutorOptions) -> Standalone {
        Standalone {
            options,
            session: ExecutionSession::new(),
        }
    }

    /// The participants of a call: the prefill's team.
    pub fn participants(&self) -> Result<usize> {
        self.options
            .participants(true, global_pool().num_workers())
            .map_err(error)
    }
}

/// Run `program` over `tokens.len()` rows, one block from position 0, on `team` participants of
/// `session`.
pub(crate) fn run_block(
    session: &mut ExecutionSession,
    team: usize,
    program: &mut Mounted,
    tokens: &[u32],
) -> Result<()> {
    if session.needs_reset() {
        return Err(error("SPMD state needs reset or restoration"));
    }
    if tokens.is_empty() || tokens.len() > program.frame_batch() {
        return Err(error(format!(
            "{} rows for a program of {}",
            tokens.len(),
            program.frame_batch()
        )));
    }
    program.frame.tokens = tokens.as_ptr();
    program.frame.total_tokens = tokens.len();
    program.frame.position = 0;
    program.frame.processed = 0;
    program.frame.head = 1;
    unsafe {
        program
            .program
            .run_spmd(global_pool(), session, &program.bindings, 1, team, None)
    }
    .map_err(error)
}

/// Room for the trace of `program` in `session`.
pub(crate) fn reserve_trace(session: &mut ExecutionSession, program: &Mounted) -> Result<()> {
    let records = program
        .program
        .sync_trace_capacity(1, global_pool().num_workers())
        .map_err(error)?;
    session.reserve_sync_trace(records).map_err(error)
}

impl Apart for Standalone {
    fn apart_spin(&self) -> Option<u32> {
        (self.options.prefill == Executor::SpmdHybrid).then_some(self.options.hybrid_spin)
    }
    fn run_apart(&mut self, program: &mut Mounted, tokens: &[u32]) -> Result<()> {
        let team = self.participants()?;
        run_block(&mut self.session, team, program, tokens)
    }
    fn reserve_apart_trace(&mut self, program: &Mounted) -> Result<()> {
        reserve_trace(&mut self.session, program)
    }
    fn apart_sync_trace(&self) -> Option<SyncTraceSnapshot> {
        self.session.sync_trace()
    }
}
