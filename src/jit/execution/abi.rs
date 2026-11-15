//! The call structures of the execution plan: the frame, the task grids, and
//! the `#[repr(C)]` records every kernel reads through its one pointer
//! argument. The lowering fills them; the x86 kernels consume them.

use super::math_data::Tables;
use super::ExecutionError;

// ── bf16 ──────────────────────────────────────────────────────────────────

/// Stable projection arguments. Input format and output width belong to the
/// compiled kernel; this context supplies only addresses and the current rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bf16Call {
    /// bf16 or f32 activations, matching the compiled input format.
    pub activations: *const u8,
    /// Existing R16 packed bf16 weights.
    pub weights: *const u16,
    /// bf16 or f32 output, matching the compiled output format.
    pub output: *mut u8,
    /// Inline rows when `row_count` is null.
    pub rows: usize,
    /// Optional rows produced by a preceding generated block node.
    pub row_count: *const usize,
}

/// Up to four projections of one input in one node: their tasks form one
/// index space, the column tiles of every part in order
/// ([`crate::jit::knobs::Knobs::fuse_projections`]).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bf16MultiCall {
    /// The calls of the parts; `count` of them are live.
    pub calls: [Bf16Call; 4],
    /// How many parts the node has.
    pub count: usize,
}

/// The packing of `A` for the matrix engine: `source` row-major `[rows, K]`
/// bf16, `dest` in blocks of sixteen rows by pairs of `K`. Rows past `rows` in
/// the last block repeat the last row: computed, never stored.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PackCall {
    /// Row-major bf16 activations.
    pub source: *const u16,
    /// The packed blocks.
    pub dest: *mut u16,
    /// Current rows, produced by block preparation.
    pub rows: *const usize,
}

// ── kv ────────────────────────────────────────────────────────────────────

/// Cache geometry fixed at mount. The calling model validates the full call's
/// absolute position range and owns the buffers for this geometry.
#[derive(Clone, Copy, Debug)]
pub struct KvSpec {
    /// KV heads.
    pub heads: usize,
    /// bf16 elements per head.
    pub head_dim: usize,
    /// bf16 elements reserved per KV head.
    pub stride: usize,
    /// Positions reserved per KV head.
    pub capacity: usize,
    /// Maximum rows in one block.
    pub max_batch: usize,
    /// AArch64: the append also writes each position into the R16 copies of
    /// `KvCall::keys_r16` and `KvCall::values_r16` (the block attention,
    /// `Knobs::attn_block`). The x86 append refuses it.
    pub r16: bool,
}
impl KvSpec {
    pub(crate) fn validate(self) -> Result<(), ExecutionError> {
        let valid = self.heads > 0
            && self.head_dim > 0
            && self.max_batch > 0
            && self.capacity >= self.max_batch
            && self
                .capacity
                .checked_mul(self.head_dim)
                .is_some_and(|n| n <= self.stride)
            && self
                .heads
                .checked_mul(self.stride)
                .and_then(|n| n.checked_mul(2))
                .is_some_and(|n| n <= isize::MAX as usize)
            && self
                .heads
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(2))
                .is_some_and(|n| n <= i32::MAX as usize)
            && self.stride <= i32::MAX as usize / 2
            // The R16 copies: whole blocks of sixteen positions and of sixteen dimensions.
            && (!self.r16 || (self.head_dim.is_multiple_of(16) && self.capacity.is_multiple_of(16)));
        if !valid {
            return Err(ExecutionError(
                "invalid generated KV reserve geometry".into(),
            ));
        }
        Ok(())
    }
}
/// Stable cache addresses. Rows and past are produced by block preparation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvCall {
    /// Head-major keys.
    pub keys: *mut u16,
    /// Head-major values.
    pub values: *mut u16,
    /// Current block's token-major projected/normalized keys.
    pub input_keys: *const u16,
    /// Current block's token-major values.
    pub input_values: *const u16,
    /// Current rows.
    pub rows: *const usize,
    /// Absolute first position of this block.
    pub past: *const usize,
    /// Under `KvSpec::r16`, per KV head (`stride` elements apart): the keys
    /// in the R16 layout with the positions as columns
    /// (`[capacity/16 blocks][head_dim/2 pairs][16][2]`), and the values
    /// transposed in the R16 layout with the dimensions as columns
    /// (`[head_dim/16 blocks][capacity/2 pairs][16][2]`) — the weights of
    /// the two GEMMs of the block attention. Null otherwise.
    pub keys_r16: *mut u16,
    /// The values' R16 copy (`Vᵀ`), as above.
    pub values_r16: *mut u16,
}

// ── attention ─────────────────────────────────────────────────────────────

/// Immutable per-layer geometry, resolved at mount. All strides here are in bf16 elements.
#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    /// Query heads.
    pub heads: usize,
    /// Query heads per KV head.
    pub gqa: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// Elements between query rows (can include the output gate).
    pub q_row: usize,
    /// Elements between query heads (can include the output gate).
    pub q_head: usize,
    /// Elements between output rows.
    pub out_row: usize,
    /// Enumerate the query heads of a KV head adjacently.
    pub grouped: bool,
    /// One task per (KV head, chunk of rows): the `gqa` query heads of the KV head run one after
    /// the other on the same participant, so that the keys and values the first one brought
    /// into its caches serve the others (`Knobs::attn_kv_task`, x86-64 prefill). The caller
    /// reserves `ceil(rows / chunk_rows) · heads / gqa` tasks.
    pub kv_task: bool,
    /// Sliding window: a row sees at most this many keys, the latest ones (itself included); 0 =
    /// every key. A causal layer's prefill takes the block kernel of the same window
    /// (`SdpaPrefillSpec::window`).
    pub window: usize,
    /// Every row sees the keys up to the end of its block (`past + rows`), from its own window's
    /// start: the soft tokens of an image see one another on a sliding layer, and the patches
    /// of the vision tower all see one another. The row kernel,
    /// or the block kernel without the causal mask (`SdpaPrefillSpec::full`: x86-64 with
    /// AVX-512 without a window, AArch64 with or without one).
    pub whole_block: bool,
    /// With `whole_block` and a `window`: the keys past the row it sees at most, `0` for every key
    /// to the end of the block. A bidirectional sliding window `|q − k| ≤ d` (EmbeddingGemma
    /// 2's sliding layers) is `window = d + 1`, `ahead = d`.
    pub ahead: usize,
    /// With `whole_block` and no `window`: the block holds several inputs end to end, and row `r`
    /// sees the keys of its own input only, `[AttentionCall::segments[2r],
    /// AttentionCall::segments[2r + 1])` (rows of the block). A block- kernel chunk whose rows
    /// straddle two inputs runs the row kernel.
    pub segments: bool,
}
/// Stable addresses and generated state for one attention operation. The caller
/// validates all buffer capacities before entry. Count/position pointers remain immutable during
/// the operation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttentionCall {
    /// Query activations.
    pub q: *const u16,
    /// Head-major KV buffers.
    pub keys: *const u16,
    /// Head-major values.
    pub values: *const u16,
    /// Token-major output.
    pub output: *mut u16,
    /// Current block rows, generated by block preparation.
    pub rows: *const usize,
    /// Absolute position of the first query in this block.
    pub past: *const usize,
    /// bf16 elements per KV head.
    pub kv_stride: usize,
    /// Attention scale.
    pub scale: f32,
    /// Softmax sink (ignored when absent from the compiled shape).
    pub sink: f32,
    /// The R16 copies of `KvCall` (`kv_stride` elements per KV head), read
    /// by the AArch64 block attention; null without it.
    pub keys_r16: *const u16,
    /// The values' R16 copy (`Vᵀ`), as above.
    pub values_r16: *const u16,
    /// [`AttentionSpec::segments`]: per row of the block, the first and the end key of its input; null without it.
    pub segments: *const u32,
}

// ── frame ─────────────────────────────────────────────────────────────────

/// Stable storage owned by a mounted model. No Rust references to fields may
/// remain live while generated code writes the frame.
#[repr(C)]
#[derive(Debug)]
pub struct CallFrame {
    /// Pointer to the caller's token IDs for this call.
    pub tokens: *const u32,
    /// Number of token IDs in this call.
    pub total_tokens: usize,
    /// Absolute model position at the current block; initialized before launch.
    pub position: usize,
    /// Tokens completed in this call; initialized to zero before launch.
    pub processed: usize,
    /// Current block length, written by `begin_block`.
    pub rows: usize,
    /// First token of the current block, written by `begin_block`.
    pub token_cursor: *const u32,
    /// One if this is the last block, zero otherwise.
    pub last_block: usize,
    /// One to run the grids of the last block (`TaskGrid::last_block_only`: the final norm and the
    /// language head), zero to skip them: a prompt fed by pieces runs the head on its last
    /// piece only. Set before launch; the RISC-V `begin_block` does not read it (the head
    /// always runs there).
    pub head: usize,
    /// Preallocated task counters, one for each mounted `TaskGrid`.
    pub counts: *mut usize,
}
/// `ceil(block_rows / rows_per_task) * tasks_per_tile`, optionally only on the
/// final block. A grid with `rows_per_task == max_batch` describes one task per
/// block (or one head operation only on the final block).
#[derive(Clone, Copy, Debug)]
pub struct TaskGrid {
    /// Rows covered by one tile, e.g. GEMM MR or SDPA prefill rows.
    pub rows_per_task: usize,
    /// Tasks per row tile, e.g. number of column tiles or attention heads.
    pub tasks_per_tile: usize,
    /// Skip this grid before the last block, e.g. final norm and language head.
    pub last_block_only: bool,
}
impl TaskGrid {
    /// Validate the grid and compute its maximum task count at mount.
    pub fn capacity(self, max_batch: usize) -> Result<usize, ExecutionError> {
        if self.rows_per_task == 0 || max_batch == 0 {
            return Err(ExecutionError("empty generated block/grid geometry".into()));
        }
        max_batch
            .div_ceil(self.rows_per_task)
            .checked_mul(self.tasks_per_tile)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| ExecutionError("generated task grid capacity overflow".into()))
    }
}

// ── delta ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GateCall {
    pub ab: *const f32,
    pub a_log: *const f32,
    pub bias: *const u16,
    pub beta: *mut f32,
    pub decay: *mut f32,
    pub tables: *const Tables,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct InterleaveCall {
    pub a: *const f32,
    pub b: *const f32,
    pub output: *mut f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct L2Call {
    pub qkv: *mut f32,
    pub eps: f32,
}
/// The context of [`l2_gates`]: the [`L2Call`] of the row's Q and K heads,
/// then the [`GateCall`] of its gates, in one word-aligned block.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct L2GatesCall {
    pub l2: L2Call,
    pub gates: GateCall,
}
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub nk: usize,
    pub nv: usize,
    pub hkd: usize,
    pub hvd: usize,
    pub ck: usize,
}
impl Shape {
    pub fn key(self) -> usize {
        self.nk * self.hkd
    }
    pub fn value(self) -> usize {
        self.nv * self.hvd
    }
    pub fn conv(self) -> usize {
        2 * self.key() + self.value()
    }
    pub fn check(self) -> Result<(), ExecutionError> {
        if self.nk == 0
            || self.nv == 0
            || !self.nv.is_multiple_of(self.nk)
            || self.hkd == 0
            || self.hvd == 0
            || self.ck < 2
            || [self.nk, self.nv, self.hkd, self.hvd, self.ck]
                .iter()
                .any(|&n| n > 1 << 16)
            || self
                .conv()
                .checked_mul(self.ck)
                .is_none_or(|n| n > i32::MAX as usize / 4)
            || self
                .nv
                .checked_mul(self.hkd)
                .and_then(|n| n.checked_mul(self.hvd))
                .is_none_or(|n| n > i32::MAX as usize / 4)
        {
            return Err(ExecutionError("invalid generated DeltaNet geometry".into()));
        }
        Ok(())
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RecurrenceCall {
    pub state: *mut f32,
    pub output: *mut f32,
    pub qkv: *const f32,
    pub beta: *const f32,
    pub decay: *const f32,
    pub rows: *const usize,
    /// The absolute position of the block's first row (`CallFrame::position`).
    pub position: *const usize,
    /// One in the prefill program, zero in the decode one: an AArch64 traversal by chunks
    /// carries a partial chunk over in prefill and closes it at decode.
    pub prefill: usize,
    /// Mamba2: the skip `D`, `[num_v_heads]` f32, passed in place of the scale; null on the
    /// DeltaNet.
    pub d: *const f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ConvCall {
    pub data: *mut f32,
    pub weights: *const u16,
    pub state: *mut f32,
    pub rows: *const usize,
    pub tables: *const Tables,
    /// Mamba2: the conv's bias, `[conv_dim]` f32; null on the DeltaNet.
    pub bias: *const f32,
}
/// What LiquidAI's gated short convolution reads and writes (`lfm2`), per row of the block: `B`,
/// `C` and `x` (f32 projections, `width` each, rounded to bf16 by the kernel as the module's bf16
/// `in_proj` is), the conv's weights (bf16, tap-major by blocks of sixteen channels), its state
/// (the last `K` values of `bf16(B · x)` per channel, f32, tap-major by blocks of sixteen), the
/// output `bf16(C · bf16(conv))`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ShortConvCall {
    pub b: *const f32,
    pub c: *const f32,
    pub x: *const f32,
    pub weights: *const u16,
    pub state: *mut f32,
    pub output: *mut u16,
    pub rows: *const usize,
}
/// What LFM3's gate per head reads and writes, per row of the block: the attention's output (`heads
/// · head_dim` bf16 a row, gated in place) and the gate's logits (`self_attn.gate_proj`, `heads`
/// bf16 a row).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HeadGateCall {
    pub output: *mut u16,
    pub gate: *const u16,
    pub rows: *const usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NormCall {
    pub output: *mut f32,
    pub gate: *const f32,
    pub weights: *const u16,
    pub output_bf16: *mut u16,
    pub tables: *const Tables,
    pub eps: f32,
}

/// The most experts a token uses.
pub const MAX_TOP_K: usize = 8;

/// How a router turns the scores of a row into the weights of its `top_k` experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteRule {
    /// Gemma 4 (`Gemma4TextRouter`): the softmax of the `top_k` best scores times
    /// `per_expert_scale`, in f32; the slots best first.
    Gemma,
    /// Qwen (`Qwen3_5MoeTopKRouter`): the softmax over every expert, the `top_k` best (the lowest
    /// expert on a tie), divided by their sum when `norm` (`norm_topk_prob`), rounded to bf16;
    /// the slots in increasing expert order, the order in which the module adds the experts
    /// up. `per_expert_scale` is not read.
    Qwen {
        /// `norm_topk_prob`.
        norm: bool,
    },
    /// LiquidAI (`Lfm2MoeTopKRouter`): `s = bf16(σ(score))` per expert, the `top_k` best of `s +
    /// bias` (f32, the bias read from `per_expert_scale`; the lowest expert on a tie), their
    /// `s` divided by `bf16(bf16(Σs) + 1e-6)` when `norm` (the sum in f32 in the order of the
    /// choice), then times the scale of `scale_bits` unless it is one, each step rounded to
    /// bf16 as the module's bf16 tensors are; the slots in increasing expert order, as Qwen's.
    Lfm {
        /// `norm_topk_prob`.
        norm: bool,
        /// `routed_scaling_factor`, its f32 bits.
        scale_bits: u32,
    },
}

/// What `route` reads and writes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RouteCall {
    /// Router scores, bf16, one per expert.
    pub scores: *const u16,
    /// `router.per_expert_scale`, f32, one per expert.
    pub per_expert_scale: *const f32,
    /// The weights of the chosen experts, f32, `top_k` of them.
    pub weights_out: *mut f32,
    /// The address of the weights of expert `e`, for the gate/up GEMMs and for the down GEMMs.
    pub gate_up_table: *const u64,
    /// Same, for the down GEMMs.
    pub down_table: *const u64,
    /// The `weights` field of the call struct of slot `k`, for the gate/up nodes and the down
    /// nodes.
    pub gate_up_slots: [*mut u64; MAX_TOP_K],
    /// Same, for the down nodes.
    pub down_slots: [*mut u64; MAX_TOP_K],
}

/// What `combine` reads and writes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CombineCall {
    /// `top_k` rows of `dim` bf16.
    pub rows: *const u16,
    /// `top_k` weights, f32.
    pub weights: *const f32,
    /// The sum, bf16.
    pub out: *mut u16,
}

/// What `route_rows` reads and writes: the router of a block of rows (the prefill of a
/// mixture-of-experts model), the rows of every expert gathered so that each expert runs one GEMM
/// over its own rows.
///
/// Per row `r` of the block (`frame.rows` of them): the `top_k` best of the `experts` scores, their
/// softmax times `per_expert_scale` into `weights_out[r·top_k + k]`, then, the rows grouped by
/// expert in expert order, the place of each `(r, k)` in that order into `pos_out[r·top_k + k]` and
/// row `r` of `source` copied there in `gathered`. For every expert `e`: `row_counts[e]` its rows,
/// the activations and output of its gate/up call (`gate_up_calls[e]`) and of its down call
/// (`down_calls[e]`) moved to its first row, and the task counts of their grids
/// (`frame.counts[gate_up_grids[e]]`, `frame.counts[down_grids[e]]`) set to `⌈rows / mr⌉ · column
/// tiles`. `total` gets `rows · top_k`, the rows of the activation and of its quantization.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RouteRowsCall {
    /// Router scores, bf16: `experts / 16` blocks of `[max_rows][16]`.
    pub scores: *const u16,
    /// `router.per_expert_scale`, f32, one per expert.
    pub per_expert_scale: *const f32,
    /// The frame of the program: the rows of the block, the task counters.
    pub frame: *const CallFrame,
    /// The weights of the chosen experts, f32, `top_k` per row.
    pub weights_out: *mut f32,
    /// The place of each `(row, k)` among the gathered rows, u32.
    pub pos_out: *mut u32,
    /// Scratch: the expert of each `(r, k)`.
    pub ids: *mut u32,
    /// The rows of each expert, read by its two calls.
    pub row_counts: *mut u64,
    /// Scratch: the next free row of each expert.
    pub cursor: *mut u64,
    /// The quantized rows the experts read (`a_row` bytes each).
    pub source: *const u8,
    /// Their copy, gathered by expert.
    pub gathered: *mut u8,
    /// `rows · top_k`.
    pub total: *mut u64,
    /// The address of the gate/up `Bf16Call` of each expert.
    pub gate_up_calls: *const u64,
    /// The address of the down `Bf16Call` of each expert.
    pub down_calls: *const u64,
    /// The task counter index of the gate/up grid of each expert.
    pub gate_up_grids: *const u64,
    /// The task counter index of the down grid of each expert.
    pub down_grids: *const u64,
    /// The first row of the outputs of the gate/up calls.
    pub gate_up_out: *mut u8,
    /// The first row of the quantized activations of the down calls.
    pub down_in: *const u8,
    /// The first row of the outputs of the down calls.
    pub down_out: *mut u8,
}

/// What `combine_rows` reads and writes, for row `r` (the task): `out[r] = Σ_k weights[r·top_k + k]
/// · rows[pos[r·top_k + k]]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CombineRowsCall {
    /// The outputs of the experts, gathered, `dim` bf16 each.
    pub rows: *const u16,
    /// The place of each `(row, k)` among them.
    pub pos: *const u32,
    /// The weight of each `(row, k)`, f32.
    pub weights: *const f32,
    /// The sums, bf16, one row per row of the block.
    pub out: *mut u16,
}

/// The geometry `route_rows` is emitted for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteRowsSpec {
    /// Experts of a layer (a multiple of 16).
    pub experts: usize,
    /// Experts a row uses.
    pub top_k: usize,
    /// Rows of a block: the stride of the score blocks.
    pub max_rows: usize,
    /// Bytes of one quantized row the gate/up calls read.
    pub a_row: usize,
    /// Bytes of one row the gate/up calls write.
    pub gate_up_out_row: usize,
    /// Bytes of one quantized row the down calls read.
    pub down_in_row: usize,
    /// Bytes of one row the down calls write.
    pub down_out_row: usize,
    /// `(mr, column tiles)` of the gate/up kernel.
    pub gate_up_tiles: (usize, usize),
    /// `(mr, column tiles)` of the down kernel.
    pub down_tiles: (usize, usize),
    /// How the weights of a row are made.
    pub rule: RouteRule,
}

/// What `qwen_moe_out` reads and writes, for row `r` (the task): the end of
/// `Qwen3_5MoeSparseMoeBlock` and the residual, every step rounded to bf16 as the module's bf16
/// tensors are: `e = Σ_k bf16(w_k · y_k)` added in slot order, each sum rounded; `g =
/// bf16(σ(bf16(shared_gate · xn[r])))`; `x[r] = bf16(x[r] + bf16(e + bf16(g · shared[r])))`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QwenOutCall {
    /// The outputs of the experts, `dim` bf16 each.
    pub rows: *const u16,
    /// The place of each `(row, k)` among them, u32 (`0..top_k` for one row).
    pub pos: *const u32,
    /// The weight of each `(row, k)`, f32 (bf16 values).
    pub weights: *const f32,
    /// The input of the block (`post_attention_layernorm(x)`), `dim` bf16 per row.
    pub xn: *const u16,
    /// `shared_expert_gate`, `dim` bf16.
    pub gate: *const u16,
    /// The shared expert's output, `dim` bf16 per row.
    pub shared: *const u16,
    /// The residual, `dim` bf16 per row, updated in place.
    pub x: *mut u16,
}
