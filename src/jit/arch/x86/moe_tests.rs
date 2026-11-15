//! The router and the combine of the mixture-of-experts block against a scalar reference (x86-64).
use super::execution::moe;
use crate::jit::execution::abi::{
    Bf16Call, CallFrame, CombineCall, CombineRowsCall, QwenOutCall, RouteCall, RouteRowsCall,
    RouteRowsSpec, RouteRule, MAX_TOP_K,
};

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn exec(bytes: &[u8]) -> memmap2::Mmap {
    crate::jit::arch::map_exec(bytes.len(), |m| m.copy_from_slice(bytes)).unwrap()
}

#[test]
fn route_matches_reference() {
    let (experts, top_k) = (128usize, 8usize);
    let mut seed = 3u64;
    for round in 0..20 {
        let mut scores: Vec<u16> = (0..experts).map(|_| bf16(rng(&mut seed) * 6.0)).collect();
        if round % 5 == 0 {
            scores[17] = scores[90];
            scores[3] = scores[90];
        }
        let pe: Vec<f32> = (0..experts).map(|_| 0.5 + rng(&mut seed).abs()).collect();
        let gu_table: Vec<u64> = (0..experts as u64).map(|e| 1000 + e).collect();
        let dn_table: Vec<u64> = (0..experts as u64).map(|e| 5000 + e).collect();
        let mut gu_slots = [0u64; MAX_TOP_K];
        let mut dn_slots = [0u64; MAX_TOP_K];
        let mut weights = [0f32; MAX_TOP_K];
        let mut call = RouteCall {
            scores: scores.as_ptr(),
            per_expert_scale: pe.as_ptr(),
            weights_out: weights.as_mut_ptr(),
            gate_up_table: gu_table.as_ptr(),
            down_table: dn_table.as_ptr(),
            gate_up_slots: [std::ptr::null_mut(); MAX_TOP_K],
            down_slots: [std::ptr::null_mut(); MAX_TOP_K],
        };
        for k in 0..top_k {
            call.gate_up_slots[k] = &mut gu_slots[k];
            call.down_slots[k] = &mut dn_slots[k];
        }
        let kernel = moe::route(experts, top_k).unwrap();
        let page = exec(&kernel.bytes);
        let run: unsafe extern "sysv64" fn(*const RouteCall) =
            unsafe { std::mem::transmute(page.as_ptr()) };
        unsafe { run(&call) };
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            f(scores[b])
                .partial_cmp(&f(scores[a]))
                .unwrap()
                .then(a.cmp(&b))
        });
        let chosen = &order[..top_k];
        let m = f(scores[chosen[0]]);
        let e: Vec<f32> = chosen.iter().map(|&i| (f(scores[i]) - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        for k in 0..top_k {
            assert_eq!(
                gu_slots[k],
                1000 + chosen[k] as u64,
                "round {round} slot {k}: expert"
            );
            assert_eq!(dn_slots[k], 5000 + chosen[k] as u64);
            let want = e[k] / sum * pe[chosen[k]];
            assert!(
                (weights[k] - want).abs() <= 2e-4 * want.abs().max(1e-3),
                "round {round} slot {k}: weight {} vs {want}",
                weights[k]
            );
        }
    }
}

#[test]
fn combine_matches_reference() {
    let (dim, top_k) = (2816usize, 8usize);
    let mut seed = 9u64;
    let rows: Vec<u16> = (0..top_k * dim).map(|_| bf16(rng(&mut seed))).collect();
    let weights: Vec<f32> = (0..8).map(|_| rng(&mut seed).abs() * 0.3).collect();
    let mut out = vec![0u16; dim];
    let call = CombineCall {
        rows: rows.as_ptr(),
        weights: weights.as_ptr(),
        out: out.as_mut_ptr(),
    };
    let kernel = moe::combine(dim, top_k).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "sysv64" fn(*const CombineCall) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    unsafe { run(&call) };
    for d in 0..dim {
        let mut acc = 0f32;
        for k in 0..top_k {
            acc = f32::mul_add(weights[k], f(rows[k * dim + d]), acc);
        }
        let got = f(out[d]);
        assert!(
            (got - acc).abs() <= 8e-3 * acc.abs().max(1e-2),
            "d {d}: {got} vs {acc}"
        );
    }
}

/// The router of a block against the router of one row, row by row: the same weights (bit for bit),
/// and the rows of every expert gathered in expert order, the calls and the task counts of each
/// expert set from them.
#[test]
fn route_rows_matches_route() {
    check_route_rows(128, RouteRule::Gemma);
}

/// The same with Qwen's rule (`qwen3_5_moe`: 256 experts), with and without `norm_topk_prob`.
#[test]
fn route_rows_qwen_matches_route() {
    check_route_rows(256, RouteRule::Qwen { norm: true });
    check_route_rows(256, RouteRule::Qwen { norm: false });
}

/// The same with LiquidAI's rule (`lfm2_moe`; the biases are `per_expert_scale`), with and without
/// `norm_topk_prob`, with a scale.
#[test]
fn route_rows_lfm_matches_route() {
    check_route_rows(
        128,
        RouteRule::Lfm {
            norm: true,
            scale_bits: 1f32.to_bits(),
        },
    );
    check_route_rows(
        96,
        RouteRule::Lfm {
            norm: false,
            scale_bits: 2.5f32.to_bits(),
        },
    );
}

fn check_route_rows(experts: usize, rule: RouteRule) {
    let (top_k, max_rows, a_row) = (8usize, 37usize, 48usize);
    let (gu_tiles, dn_tiles) = ((4usize, 88usize), (4usize, 176usize));
    let (gu_out_row, dn_in_row, dn_out_row) = (24usize, 40usize, 56usize);
    let single = moe::route_with(experts, top_k, rule).unwrap();
    let single_page = exec(&single.bytes);
    let single_run: unsafe extern "sysv64" fn(*const RouteCall) =
        unsafe { std::mem::transmute(single_page.as_ptr()) };
    let spec = RouteRowsSpec {
        experts,
        top_k,
        max_rows,
        a_row,
        gate_up_out_row: gu_out_row,
        down_in_row: dn_in_row,
        down_out_row: dn_out_row,
        gate_up_tiles: gu_tiles,
        down_tiles: dn_tiles,
        rule,
    };
    let kernel = moe::route_rows(spec).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "sysv64" fn(*const RouteRowsCall) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let mut seed = 5u64;
    for rows in [1usize, 2, 17, max_rows] {
        // Row r, expert e at block e/16, `[max_rows][16]`.
        let mut row_scores: Vec<Vec<u16>> = (0..rows)
            .map(|_| (0..experts).map(|_| bf16(rng(&mut seed) * 6.0)).collect())
            .collect();
        if rows > 1 {
            row_scores[1][5] = row_scores[1][77];
        }
        let mut scores = vec![0u16; experts / 16 * max_rows * 16];
        for (r, row) in row_scores.iter().enumerate() {
            for (e, &v) in row.iter().enumerate() {
                scores[(e / 16) * max_rows * 16 + r * 16 + e % 16] = v;
            }
        }
        let pe: Vec<f32> = (0..experts).map(|_| 0.5 + rng(&mut seed).abs()).collect();
        let source: Vec<u8> = (0..rows * a_row).map(|i| (i * 7 + 3) as u8).collect();
        let mut gathered = vec![0u8; rows * top_k * a_row];
        let mut weights = vec![0f32; rows * top_k];
        let mut pos = vec![0u32; rows * top_k];
        let mut ids = vec![0u32; rows * top_k];
        let mut row_counts = vec![99u64; experts];
        let mut cursor = vec![0u64; experts];
        let mut total = 0u64;
        let mut gu_calls: Vec<Bf16Call> = (0..experts)
            .map(|_| Bf16Call {
                activations: std::ptr::null(),
                weights: std::ptr::null(),
                output: std::ptr::null_mut(),
                rows: 0,
                row_count: std::ptr::null(),
            })
            .collect();
        let mut dn_calls = gu_calls.clone();
        let gu_table: Vec<u64> = gu_calls
            .iter_mut()
            .map(|c| c as *mut Bf16Call as u64)
            .collect();
        let dn_table: Vec<u64> = dn_calls
            .iter_mut()
            .map(|c| c as *mut Bf16Call as u64)
            .collect();
        // Grid e of the gate/up calls at counter 3 + e, of the down calls at 3 + experts + e.
        let gu_grids: Vec<u64> = (0..experts as u64).map(|e| 3 + e).collect();
        let dn_grids: Vec<u64> = (0..experts as u64)
            .map(|e| 3 + experts as u64 + e)
            .collect();
        let mut counts = vec![12345usize; 3 + 2 * experts + 1];
        let frame = CallFrame {
            tokens: std::ptr::null(),
            total_tokens: 0,
            position: 0,
            processed: 0,
            rows,
            token_cursor: std::ptr::null(),
            last_block: 0,
            head: 1,
            counts: counts.as_mut_ptr(),
        };
        let (gu_out, dn_in, dn_out) = (0x10_0000usize, 0x20_0000usize, 0x30_0000usize);
        let call = RouteRowsCall {
            scores: scores.as_ptr(),
            per_expert_scale: pe.as_ptr(),
            frame: &frame,
            weights_out: weights.as_mut_ptr(),
            pos_out: pos.as_mut_ptr(),
            ids: ids.as_mut_ptr(),
            row_counts: row_counts.as_mut_ptr(),
            cursor: cursor.as_mut_ptr(),
            source: source.as_ptr(),
            gathered: gathered.as_mut_ptr(),
            total: &mut total,
            gate_up_calls: gu_table.as_ptr(),
            down_calls: dn_table.as_ptr(),
            gate_up_grids: gu_grids.as_ptr(),
            down_grids: dn_grids.as_ptr(),
            gate_up_out: gu_out as *mut u8,
            down_in: dn_in as *const u8,
            down_out: dn_out as *mut u8,
        };
        unsafe { run(&call) };
        assert_eq!(total, (rows * top_k) as u64);
        // Each row as the single-row router sees it.
        let mut chosen = vec![0usize; rows * top_k];
        for r in 0..rows {
            let gu_t: Vec<u64> = (0..experts as u64).collect();
            let mut gu_slots = [0u64; MAX_TOP_K];
            let mut dn_slots = [0u64; MAX_TOP_K];
            let mut w = [0f32; MAX_TOP_K];
            let mut one = RouteCall {
                scores: row_scores[r].as_ptr(),
                per_expert_scale: pe.as_ptr(),
                weights_out: w.as_mut_ptr(),
                gate_up_table: gu_t.as_ptr(),
                down_table: gu_t.as_ptr(),
                gate_up_slots: [std::ptr::null_mut(); MAX_TOP_K],
                down_slots: [std::ptr::null_mut(); MAX_TOP_K],
            };
            for k in 0..top_k {
                one.gate_up_slots[k] = &mut gu_slots[k];
                one.down_slots[k] = &mut dn_slots[k];
            }
            unsafe { single_run(&one) };
            for k in 0..top_k {
                assert_eq!(
                    ids[r * top_k + k] as u64,
                    gu_slots[k],
                    "rows {rows} row {r} slot {k}: expert"
                );
                assert_eq!(
                    weights[r * top_k + k].to_bits(),
                    w[k].to_bits(),
                    "rows {rows} row {r} slot {k}: weight"
                );
                chosen[r * top_k + k] = gu_slots[k] as usize;
            }
        }
        // The rows of every expert, in expert order then row order.
        let mut first = 0usize;
        let mut want_pos = vec![0u32; rows * top_k];
        for e in 0..experts {
            let mine: Vec<usize> = (0..rows * top_k).filter(|&i| chosen[i] == e).collect();
            assert_eq!(
                row_counts[e],
                mine.len() as u64,
                "rows {rows} expert {e}: rows"
            );
            for (j, &i) in mine.iter().enumerate() {
                want_pos[i] = (first + j) as u32;
            }
            let c = mine.len();
            assert_eq!(
                gu_calls[e].activations as usize,
                gathered.as_ptr() as usize + first * a_row
            );
            assert_eq!(gu_calls[e].output as usize, gu_out + first * gu_out_row);
            assert_eq!(dn_calls[e].activations as usize, dn_in + first * dn_in_row);
            assert_eq!(dn_calls[e].output as usize, dn_out + first * dn_out_row);
            assert_eq!(
                counts[3 + e],
                c.div_ceil(gu_tiles.0) * gu_tiles.1,
                "rows {rows} expert {e}: gate/up tasks"
            );
            assert_eq!(
                counts[3 + experts + e],
                c.div_ceil(dn_tiles.0) * dn_tiles.1,
                "rows {rows} expert {e}: down tasks"
            );
            first += c;
        }
        assert_eq!(first, rows * top_k);
        assert_eq!(pos, want_pos, "rows {rows}: places");
        for i in 0..rows * top_k {
            let r = i / top_k;
            let p = pos[i] as usize;
            assert_eq!(
                &gathered[p * a_row..(p + 1) * a_row],
                &source[r * a_row..(r + 1) * a_row],
                "rows {rows} (row {r}, slot {}): copy",
                i % top_k
            );
        }
        assert_eq!(counts[0], 12345);
        assert_eq!(counts[3 + 2 * experts], 12345);
    }
}

/// The weighted sum of one row of a block against the one-row combine, bit for bit.
#[test]
fn combine_rows_matches_combine() {
    let (dim, top_k, rows) = (2816usize, 8usize, 5usize);
    let mut seed = 11u64;
    let gathered: Vec<u16> = (0..rows * top_k * dim)
        .map(|_| bf16(rng(&mut seed)))
        .collect();
    let weights: Vec<f32> = (0..rows * top_k)
        .map(|_| rng(&mut seed).abs() * 0.3)
        .collect();
    // A permutation of the gathered rows.
    let mut pos: Vec<u32> = (0..(rows * top_k) as u32).collect();
    for i in (1..pos.len()).rev() {
        let j = ((rng(&mut seed).abs() * (i + 1) as f32) as usize).min(i);
        pos.swap(i, j);
    }
    let mut out = vec![0u16; rows * dim];
    let call = CombineRowsCall {
        rows: gathered.as_ptr(),
        pos: pos.as_ptr(),
        weights: weights.as_ptr(),
        out: out.as_mut_ptr(),
    };
    let kernel = moe::combine_rows(dim, top_k).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "sysv64" fn(*const CombineRowsCall, usize) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let one = moe::combine(dim, top_k).unwrap();
    let one_page = exec(&one.bytes);
    let one_run: unsafe extern "sysv64" fn(*const CombineCall) =
        unsafe { std::mem::transmute(one_page.as_ptr()) };
    for r in 0..rows {
        unsafe { run(&call, r) };
        let picked: Vec<u16> = (0..top_k)
            .flat_map(|k| gathered[pos[r * top_k + k] as usize * dim..][..dim].to_vec())
            .collect();
        let mut want = vec![0u16; dim];
        let c = CombineCall {
            rows: picked.as_ptr(),
            weights: weights[r * top_k..].as_ptr(),
            out: want.as_mut_ptr(),
        };
        unsafe { one_run(&c) };
        assert_eq!(&out[r * dim..(r + 1) * dim], &want[..], "row {r}");
    }
}

/// bf16 rounding to nearest even, kept as f32.
fn rne(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// Qwen's route (`Qwen3_5MoeTopKRouter`) against a scalar reference: the experts, in increasing
/// order, and their weights to one bf16 step (the emitted exponential is a polynomial).
#[test]
fn route_qwen_matches_reference() {
    let (experts, top_k) = (256usize, 8usize);
    let mut seed = 21u64;
    for norm in [true, false] {
        let kernel = moe::route_with(experts, top_k, RouteRule::Qwen { norm }).unwrap();
        let page = exec(&kernel.bytes);
        let run: unsafe extern "sysv64" fn(*const RouteCall) =
            unsafe { std::mem::transmute(page.as_ptr()) };
        let (mut exact, mut all) = (0usize, 0usize);
        for round in 0..200 {
            let mut scores: Vec<u16> = (0..experts).map(|_| bf16(rng(&mut seed) * 4.0)).collect();
            if round % 5 == 0 {
                scores[17] = scores[200];
                scores[3] = scores[200];
            }
            let gu_table: Vec<u64> = (0..experts as u64).map(|e| 1000 + e).collect();
            let dn_table: Vec<u64> = (0..experts as u64).map(|e| 5000 + e).collect();
            let mut gu_slots = [0u64; MAX_TOP_K];
            let mut dn_slots = [0u64; MAX_TOP_K];
            let mut weights = [0f32; MAX_TOP_K];
            let mut call = RouteCall {
                scores: scores.as_ptr(),
                per_expert_scale: std::ptr::null(),
                weights_out: weights.as_mut_ptr(),
                gate_up_table: gu_table.as_ptr(),
                down_table: dn_table.as_ptr(),
                gate_up_slots: [std::ptr::null_mut(); MAX_TOP_K],
                down_slots: [std::ptr::null_mut(); MAX_TOP_K],
            };
            for k in 0..top_k {
                call.gate_up_slots[k] = &mut gu_slots[k];
                call.down_slots[k] = &mut dn_slots[k];
            }
            unsafe { run(&call) };
            let m = scores
                .iter()
                .map(|&b| f(b))
                .fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|&b| (f(b) - m).exp()).collect();
            let total: f32 = e.iter().sum();
            let p: Vec<f32> = e.iter().map(|v| v / total).collect();
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| {
                f(scores[b])
                    .partial_cmp(&f(scores[a]))
                    .unwrap()
                    .then(a.cmp(&b))
            });
            let mut chosen: Vec<usize> = order[..top_k].to_vec();
            let sum: f32 = chosen.iter().map(|&i| p[i]).sum();
            chosen.sort();
            for k in 0..top_k {
                assert_eq!(
                    gu_slots[k],
                    1000 + chosen[k] as u64,
                    "norm {norm} round {round} slot {k}: expert"
                );
                assert_eq!(dn_slots[k], 5000 + chosen[k] as u64);
                let want = rne(if norm {
                    p[chosen[k]] / sum
                } else {
                    p[chosen[k]]
                });
                assert_eq!(weights[k].to_bits() & 0xffff, 0, "a bf16 weight");
                assert!(
                    (weights[k] - want).abs() <= want.abs() / 128.0 + 1e-30,
                    "norm {norm} round {round} slot {k}: weight {} vs {want}",
                    weights[k]
                );
                exact += usize::from(weights[k] == want);
                all += 1;
            }
        }
        assert!(
            exact * 100 >= all * 98,
            "norm {norm}: {exact} of {all} weights exact"
        );
    }
}

/// The end of Qwen's sparse block and the residual against the module's arithmetic in scalar f32,
/// every step rounded to bf16: equal but for a step of bf16 on a few values (the polynomial
/// exponential of the gate).
#[test]
fn qwen_out_matches_reference() {
    let (dim, top_k, rows) = (2048usize, 8usize, 4usize);
    let mut seed = 31u64;
    let kernel = moe::qwen_out(dim, top_k).unwrap();
    let page = exec(&kernel.bytes);
    let run: unsafe extern "sysv64" fn(*const QwenOutCall, usize) =
        unsafe { std::mem::transmute(page.as_ptr()) };
    let gathered: Vec<u16> = (0..rows * top_k * dim)
        .map(|_| bf16(rng(&mut seed) * 0.5))
        .collect();
    let weights: Vec<f32> = (0..rows * top_k)
        .map(|_| rne(rng(&mut seed).abs() * 0.3))
        .collect();
    let mut pos: Vec<u32> = (0..(rows * top_k) as u32).collect();
    for i in (1..pos.len()).rev() {
        let j = ((rng(&mut seed).abs() * (i + 1) as f32) as usize).min(i);
        pos.swap(i, j);
    }
    let xn: Vec<u16> = (0..rows * dim).map(|_| bf16(rng(&mut seed))).collect();
    let gate: Vec<u16> = (0..dim).map(|_| bf16(rng(&mut seed) * 0.05)).collect();
    let shared: Vec<u16> = (0..rows * dim)
        .map(|_| bf16(rng(&mut seed) * 0.5))
        .collect();
    let x0: Vec<u16> = (0..rows * dim)
        .map(|_| bf16(rng(&mut seed) * 4.0))
        .collect();
    let mut x = x0.clone();
    let call = QwenOutCall {
        rows: gathered.as_ptr(),
        pos: pos.as_ptr(),
        weights: weights.as_ptr(),
        xn: xn.as_ptr(),
        gate: gate.as_ptr(),
        shared: shared.as_ptr(),
        x: x.as_mut_ptr(),
    };
    let (mut exact, mut off) = (0usize, 0usize);
    for r in 0..rows {
        unsafe { run(&call, r) };
        let logit: f32 = (0..dim).map(|d| f(xn[r * dim + d]) * f(gate[d])).sum();
        let g = rne(1.0 / (1.0 + (-rne(logit)).exp()));
        for d in 0..dim {
            let mut acc = 0f32;
            for k in 0..top_k {
                let y = f(gathered[pos[r * top_k + k] as usize * dim + d]);
                acc = rne(acc + rne(y * weights[r * top_k + k]));
            }
            let o = rne(acc + rne(g * f(shared[r * dim + d])));
            let want = rne(f(x0[r * dim + d]) + o);
            let got = f(x[r * dim + d]);
            if got == want {
                exact += 1;
            } else {
                assert!(
                    (got - want).abs() <= want.abs() / 64.0 + 1e-6,
                    "row {r} d {d}: {got} vs {want}"
                );
                off += 1;
            }
        }
    }
    assert!(
        off * 100 <= (exact + off),
        "{off} of {} values a step off",
        exact + off
    );
}
