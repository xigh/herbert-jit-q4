//! The one decode-time decision the lowering asks about, read from the
//! installed knobs: whether a node at `m = 1` takes the whole team.

/// The pool at `m = 1`: projections, SDPA, DeltaNet.
#[derive(Clone, Copy, Debug)]
pub struct OptionsDecode {
    /// Whole team at `m = 1` ([`crate::jit::knobs::Knobs::decode_whole_team`]).
    pub pool_m1: bool,
}

/// The decode options in force.
pub fn decode() -> OptionsDecode {
    OptionsDecode {
        pool_m1: crate::jit::knobs::knobs().decode_whole_team,
    }
}

/// `HERBERT_JIT_PLAN=1`: every kernel compiled (GEMM, attention, elementwise, DeltaNet) on stderr
/// once, with what it was given and why a shape was declined (`DEBUG [backend_jit] …`).
pub fn debug_plan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("HERBERT_JIT_PLAN").is_ok_and(|v| !v.is_empty() && v != "0"))
}
