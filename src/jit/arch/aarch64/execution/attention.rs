//! Generated SDPA head/chunk traversal: the causal bound of every row and
//! the partial prefill chunks. Calculation never calls the Rust SDPA driver.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::aarch64::util::{exec_err, madd_imm, mul_imm, Frame};
pub use crate::jit::execution::abi::{AttentionCall, AttentionSpec};
use crate::jit::sdpa::{CompiledSdpa, CompiledSdpaPrefill};
use a64::*;
use std::mem::offset_of;

/// The decode entry as one relocatable kernel with the `SdpaHeadFn`
/// signature, including the three-call Split form chained here.
pub fn head_kernel(sdpa: &CompiledSdpa) -> Result<GeneratedKernel, ExecutionError> {
    let blobs = sdpa.lowering_blobs();
    if blobs.len() == 1 {
        return unsafe { GeneratedKernel::new(blobs.into_iter().next().unwrap()) };
    }
    let mut a = Asm::new();
    let labels: Vec<Label> = blobs.iter().map(|_| a.label()).collect();
    // The six integer arguments and the sink live on the stack across calls.
    let frame = Frame::enter(&mut a, true, &[], false, 64);
    a.stp(X0, X1, mem(SP, 0));
    a.stp(X2, X3, mem(SP, 16));
    a.stp(X4, X5, mem(SP, 32));
    a.str(s(1), mem(SP, 48));
    // QK(q, k, scores, n, scale).
    a.mov(X2, X3);
    a.mov(X3, X5);
    a.bl(labels[0]);
    // Softmax(scores, n, sink).
    a.ldr(X0, mem(SP, 24));
    a.ldr(X1, mem(SP, 40));
    a.ldr(s(0), mem(SP, 48));
    a.bl(labels[1]);
    // SV(scores, v, out, n).
    a.ldr(X0, mem(SP, 24));
    a.ldr(X1, mem(SP, 16));
    a.ldr(X2, mem(SP, 32));
    a.ldr(X3, mem(SP, 40));
    a.bl(labels[2]);
    frame.leave(&mut a);
    a.ret();
    for (blob, &label) in blobs.iter().zip(&labels) {
        a.align(16);
        a.bind(label);
        a.bytes(blob);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit `(call: *const AttentionCall, task: usize, worker_scores: *mut f32)`.
/// The caller reserves `ceil(rows / chunk_rows) * heads` tasks. Chunk rows are
/// those of the supplied prefill kernel (the sixteen of `sdpa_block.rs`,
/// which also receives the head's R16 copies in `x6`/`x7`), or one without
/// it.
///
/// The block kernel's tiles sit on multiples of sixteen positions, not on
/// the first row of the pass: a row's bits are then the same wherever a
/// prefill call starts (as on x86-64, whose block kernel has the bits of the
/// decode kernel). The pass covers `ceil((past % 16 + rows) / 16)` tiles, one
/// more than its chunks at most; chunk `c` runs tiles `c`, `c + chunks`, …,
/// so the first chunk of a head takes the extra one. A tile the pass covers
/// in part runs the block kernel too, on its rows only.
pub fn tile_selector(
    sdpa: &CompiledSdpa,
    prefill: Option<&CompiledSdpaPrefill>,
    spec: AttentionSpec,
) -> Result<GeneratedKernel, ExecutionError> {
    if spec.heads == 0
        || spec.gqa == 0
        || !spec.heads.is_multiple_of(spec.gqa)
        || spec.head_dim != sdpa.spec().shape.head_dim
        || spec.q_head < spec.head_dim
        || spec
            .heads
            .checked_mul(spec.q_head)
            .is_none_or(|n| n > spec.q_row)
        || spec
            .heads
            .checked_mul(spec.head_dim)
            .is_none_or(|n| n > spec.out_row)
        || [
            spec.heads,
            spec.gqa,
            spec.head_dim,
            spec.q_row,
            spec.q_head,
            spec.out_row,
        ]
        .iter()
        .any(|&n| n > i32::MAX as usize / 2)
    {
        return Err(ExecutionError(
            "invalid generated attention geometry".into(),
        ));
    }
    if let Some(p) = prefill {
        if p.spec().shape != sdpa.spec().shape
            || p.spec().row_stride != spec.q_row
            || p.spec().out_row_stride != spec.out_row
        {
            return Err(ExecutionError(
                "prefill attention kernel incompatible with layer policy".into(),
            ));
        }
    }
    if spec.ahead > 0 && (spec.window == 0 || !spec.whole_block) {
        return Err(ExecutionError(
            "`ahead` bounds the keys of a whole-block row in a window".into(),
        ));
    }
    if spec.segments && (!spec.whole_block || spec.window > 0) {
        return Err(ExecutionError(
            "`segments`: a whole-block layer without a window".into(),
        ));
    }
    // A sliding layer takes the block kernel of its window (`SdpaPrefillSpec::window`): the tile's
    // absolute `past` goes in as for a causal one, the kernel moves past the keys before the
    // window. A whole-block layer takes the block kernel without the causal mask
    // (`SdpaPrefillSpec::full`); with a window, the bidirectional one only (`ahead = window −
    // 1`, `|q − k| < window`).
    if prefill.is_some_and(|p| p.spec().full != spec.whole_block) {
        return Err(ExecutionError(
            "a whole-block layer takes the block kernel without the causal mask, and only it"
                .into(),
        ));
    }
    if prefill.is_some() && spec.whole_block && spec.window > 0 && spec.ahead + 1 != spec.window {
        return Err(ExecutionError(
            "the unmasked block kernel's window is `|q − k| < window`".into(),
        ));
    }
    if prefill.is_some_and(|p| p.spec().window != spec.window) {
        return Err(ExecutionError(
            "a sliding-window layer takes the block kernel of its window".into(),
        ));
    }
    let chunk_rows = prefill.map_or(1, CompiledSdpaPrefill::rows);
    let head = head_kernel(sdpa)?;
    let block = prefill.map(CompiledSdpaPrefill::lowering_bytes);
    emit_selector(spec, chunk_rows, head, block)
}

// x19 call, x20 query head, x21 row, x22 end row, x23 KV head, x26 worker scores;
// with the block kernel, x22 the tile's first row, x24 the tile, x25 the tiles
// of the pass, x27 the chunks, x28 the tile's end row.
fn emit_selector(
    spec: AttentionSpec,
    chunk_rows: usize,
    head: GeneratedKernel,
    block: Option<Vec<u8>>,
) -> Result<GeneratedKernel, ExecutionError> {
    if block.is_some() && chunk_rows != 16 {
        return Err(ExecutionError(
            "the block attention kernel serves tiles of sixteen positions".into(),
        ));
    }
    let mut a = Asm::new();
    let head_label = a.label();
    let block_label = a.label();
    let exit = a.label();
    let row_loop = a.label();
    let field = |f: usize| mem(X19, f as i64);
    let frame = if block.is_some() {
        Frame::enter(
            &mut a,
            true,
            &[X19, X20, X21, X22, X23, X24, X25, X26, X27, X28],
            false,
            0,
        )
    } else {
        Frame::enter(&mut a, true, &[X19, X20, X21, X22, X23, X26], false, 0)
    };
    a.mov(X19, X0);
    a.mov(X26, X2);
    a.ldr(X8, field(offset_of!(AttentionCall, rows)));
    a.ldr(X8, at(X8));
    a.cbz(X8, exit);
    // chunks = ceil(rows / chunk_rows)
    a.mov_imm(X11, chunk_rows as u64);
    a.add(X9, X8, X11);
    a.sub(X9, X9, 1);
    a.udiv(X9, X9, X11);
    if spec.grouped {
        a.mov_imm(X10, spec.gqa as u64);
        a.udiv(X12, X1, X10);
        a.msub(X20, X12, X10, X1); // q head within the KV group
        a.udiv(X13, X12, X9); // KV head
        a.msub(X12, X13, X9, X12); // chunk
        madd_imm(&mut a, X20, X13, spec.gqa as u64, X20, X14);
    } else {
        a.udiv(X20, X1, X9);
        a.msub(X12, X20, X9, X1);
    }
    a.mov_imm(X10, spec.heads as u64);
    a.cmp(X20, X10);
    a.bhs(exit);
    a.mov_imm(X10, spec.gqa as u64);
    a.udiv(X23, X20, X10);
    if block.is_some() {
        // x24 = tile c, x27 = chunks, x25 = tiles = ceil((past % 16 + rows) / 16).
        a.mov(X24, X12);
        a.mov(X27, X9);
        a.ldr(X13, field(offset_of!(AttentionCall, past)));
        a.ldr(X13, at(X13));
        a.and(X10, X13, 15u64);
        a.add(X25, X10, X8);
        a.add(X25, X25, 15);
        a.lsr(X25, X25, 4);
        let tiles = a.label();
        a.bind(tiles);
        a.cmp(X24, X25);
        a.bhs(exit);
        // x10 = the tile's first position, `past` rounded down plus 16 per tile;
        // its rows of the pass, from max(x10, past) to min(x10 + 16, past + rows).
        a.ldr(X13, field(offset_of!(AttentionCall, past)));
        a.ldr(X13, at(X13));
        a.ldr(X14, field(offset_of!(AttentionCall, rows)));
        a.ldr(X14, at(X14));
        a.add(X14, X14, X13);
        a.and(X10, X13, !15u64);
        a.add(X10, X10, X24.lsl(4));
        a.cmp(X10, X13);
        a.csel(X11, X10, X13, Cond::Hi);
        a.add(X12, X10, chunk_rows as u64);
        a.cmp(X12, X14);
        a.csel(X12, X14, X12, Cond::Hi);
        // x21 = the first row in the pass; x22 and x28 the first and the end rows in the tile.
        a.sub(X21, X11, X13);
        a.sub(X22, X11, X10);
        a.sub(X28, X12, X10);
        kv_offset(&mut a);
        // x5 = the keys before the tile.
        a.ldr(X5, field(offset_of!(AttentionCall, past)));
        a.ldr(X5, at(X5));
        a.add(X5, X5, X21);
        a.sub(X5, X5, X22);
        // The head's R16 copies, at the same offset as its cache.
        a.ldr(X6, field(offset_of!(AttentionCall, keys_r16)));
        a.add(X6, X6, X10);
        a.ldr(X7, field(offset_of!(AttentionCall, values_r16)));
        a.add(X7, X7, X10);
        let straddle = a.label();
        if spec.whole_block {
            // Without the mask, x15 = the end of the keys, x17 = the first (absolute): the input of
            // the tile's rows (a tile across two inputs runs the row kernel), or the
            // whole block.
            if spec.segments {
                a.ldr(X14, field(offset_of!(AttentionCall, segments)));
                a.add(X13, X14, X21.lsl(3));
                a.ldr(w(17), at(X13));
                a.ldr(w(15), mem(X13, 4));
                // The tile's last row of the pass: x21 + (x28 − x22) − 1.
                a.sub(X12, X28, X22);
                a.add(X12, X12, X21);
                a.add(X12, X14, X12.lsl(3));
                a.ldr(w(12), mem(X12, -8));
                a.cmp(X12, X17);
                a.bne(straddle);
            } else {
                block_end(&mut a, X15);
                a.mov_imm(X17, 0);
            }
        }
        arguments(&mut a, spec);
        if spec.whole_block {
            a.mov(X1, X15);
            a.mov(X2, X17);
        }
        a.mov(X8, X22);
        a.mov(X9, X28);
        a.bl(block_label);
        a.add(X24, X24, X27);
        a.b(tiles);
        if spec.segments {
            // The tile's rows of the pass one by one, then the next tile.
            a.bind(straddle);
            a.sub(X12, X28, X22);
            a.add(X22, X21, X12);
            let top = a.label();
            a.bind(top);
            row_call(&mut a, spec, head_label);
            a.add(X21, X21, 1);
            a.cmp(X21, X22);
            a.blo(top);
            a.add(X24, X24, X27);
            a.b(tiles);
        } else {
            a.bind(straddle);
        }
    }
    mul_imm(&mut a, X21, X12, chunk_rows as u64, X14);
    a.add(X22, X21, X11);
    a.cmp(X22, X8);
    a.csel(X22, X8, X22, Cond::Hi);
    a.bind(row_loop);
    row_call(&mut a, spec, head_label);
    a.add(X21, X21, 1);
    a.cmp(X21, X22);
    a.blo(row_loop);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(head_label);
    a.bytes(&head.bytes);
    if let Some(blob) = block {
        a.align(16);
        a.bind(block_label);
        a.bytes(&blob);
    } else {
        a.bind(block_label);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// The row kernel on row x21 of query head x20 (KV head x23): its bounds, then the call. Clobbers
/// x0..x17.
fn row_call(a: &mut Asm, spec: AttentionSpec, head_label: Label) {
    let field = |f: usize| mem(X19, f as i64);
    kv_offset(a);
    // x5 = visible keys: the causal past plus the row itself.
    a.ldr(X5, field(offset_of!(AttentionCall, past)));
    a.ldr(X5, at(X5));
    a.add(X5, X5, X21);
    a.add(X5, X5, 1);
    if spec.window > 0 {
        // A sliding window: the last `window` keys. x11 = min(n, window), x12 = the first key seen,
        // the key and value pointers move by that many rows.
        a.mov_imm(X11, spec.window as u64);
        a.cmp(X5, X11);
        a.csel(X11, X5, X11, Cond::Lo);
        a.sub(X12, X5, X11);
        madd_imm(a, X10, X12, (2 * spec.head_dim) as u64, X10, X13);
        if spec.whole_block {
            // The keys up to the end of the block, from the row's window start (x12); with `ahead`,
            // up to `ahead` keys past the row at most (x15).
            if spec.ahead > 0 {
                a.mov_imm(X15, spec.ahead as u64);
                a.add(X15, X5, X15);
            }
            block_end(a, X5);
            if spec.ahead > 0 {
                a.cmp(X5, X15);
                a.csel(X5, X15, X5, Cond::Hi);
            }
            a.sub(X5, X5, X12);
        } else {
            a.mov(X5, X11);
        }
    } else if spec.segments {
        // The keys of the row's input only: x13 its first, x5 its end, then their count; x10 moved
        // to the first.
        a.ldr(X14, field(offset_of!(AttentionCall, segments)));
        a.add(X14, X14, X21.lsl(3));
        a.ldr(w(13), at(X14));
        a.ldr(w(5), mem(X14, 4));
        a.sub(X5, X5, X13);
        madd_imm(a, X10, X13, (2 * spec.head_dim) as u64, X10, X12);
    } else if spec.whole_block {
        block_end(a, X5);
    }
    arguments(a, spec);
    a.bl(head_label);
}

/// `reg` = the end of the block, `past + rows` (x13, x14 clobbered): the keys a whole-block row
/// sees.
fn block_end(a: &mut Asm, reg: R) {
    a.ldr(X13, mem(X19, offset_of!(AttentionCall, rows) as i64));
    a.ldr(X13, at(X13));
    a.ldr(X14, mem(X19, offset_of!(AttentionCall, past) as i64));
    a.ldr(X14, at(X14));
    a.add(reg, X13, X14);
}

/// `x10` = byte offset of the KV head in each buffer: every cache starts at position 0.
fn kv_offset(a: &mut Asm) {
    a.ldr(X10, mem(X19, offset_of!(AttentionCall, kv_stride) as i64));
    a.mul(X10, X10, X23);
    a.lsl(X10, X10, 1);
}

// x10 = byte offset in each KV buffer, x5 = n or past. The head kernels may
// clobber every argument register; the live state is callee-saved.
fn arguments(a: &mut Asm, spec: AttentionSpec) {
    let field = |f: usize| mem(X19, f as i64);
    a.ldr(X1, field(offset_of!(AttentionCall, keys)));
    a.add(X1, X1, X10);
    a.ldr(X2, field(offset_of!(AttentionCall, values)));
    a.add(X2, X2, X10);
    a.mov(X3, X26);
    a.ldr(X0, field(offset_of!(AttentionCall, q)));
    mul_imm(a, X10, X21, spec.q_row as u64, X12);
    madd_imm(a, X10, X20, spec.q_head as u64, X10, X12);
    a.add(X0, X0, X10.lsl(1));
    a.ldr(X4, field(offset_of!(AttentionCall, output)));
    mul_imm(a, X10, X21, spec.out_row as u64, X12);
    madd_imm(a, X10, X20, spec.head_dim as u64, X10, X12);
    a.add(X4, X4, X10.lsl(1));
    a.ldr(s(0), field(offset_of!(AttentionCall, scale)));
    a.ldr(s(1), field(offset_of!(AttentionCall, sink)));
}
