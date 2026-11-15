//! `HERBERT_PREFILL_PROFILE=1`: where the time of a text prefill goes, by operator and by layer.
//!
//! The prefill regime is then mounted with the synchronisation stamps (`Measure::sync_trace`,
//! emitted `rdtscp` on x86-64, `cntvct_el0` and the core number of `tpidr_el0` on AArch64,
//! around every node and its barrier, for every participant); the decode regime is emitted as
//! without them. After every call of the prefill program (a prompt fed in pieces makes several),
//! the trace of that call is folded here; at the end of the prefill, one report on stderr and one
//! JSON line (`[prefill-profile-json] {...}`).
//!
//! Per node and block: its wall time runs from the last departure from the previous barrier to the
//! last departure from its own (as `print_profile_as`), the stamps scaled to the call's wall time;
//! its work time is the sum over participants of `work_end - work_start`, its wait the sum of
//! `barrier_end - barrier_start` (spin and sleep). A stamp left at zero (a path not taken) counts
//! for nothing.
//!
//! Categories: `attention` (tiled attention and the K/V append), `gemm` (the emitted projections,
//! the experts, the language head, and the packing / quantisation of their activations),
//! `recurrent` (DeltaNet, Mamba, LFM's short convolution), `elem` (norms, RoPE, activations, MoE
//! routing and combination, embedding, argmax), `other` (any label this table does not know: it is
//! printed by name). Inside `gemm`, the projections are told apart by their place in the layer:
//! before the mixer (`qkv`: Q/K/V, or the in-projections of a recurrent layer), the first after
//! it (`o`), the one before a MoE route (`router`), the rest (`mlp`). This is read from the plan's
//! order, not from the weights.
use crate::jit::execution::{NodeLabel, SyncTraceSnapshot};
use std::fmt::Write as _;

/// The sub-categories, in print order, and the category each belongs to.
const SUBS: [(&str, &str); 13] = [
    ("attention", "attention"),
    ("gemm-qkv", "gemm"),
    ("gemm-o", "gemm"),
    ("gemm-mlp", "gemm"),
    ("gemm-router", "gemm"),
    ("gemm-experts", "gemm"),
    ("gemm-lm-head", "gemm"),
    ("gemm-prep", "gemm"),
    ("recurrent", "recurrent"),
    ("elem", "elem"),
    ("moe-routing", "elem"),
    ("block", "elem"),
    ("other", "other"),
];
const CATS: [&str; 5] = ["attention", "gemm", "recurrent", "elem", "other"];

fn is_mixer(op: &str) -> bool {
    matches!(
        op,
        "attention" | "delta-causal-recurrence" | "short-convolution" | "mamba-causal-recurrence"
    )
}

fn is_projection(op: &str) -> bool {
    matches!(op, "bf16-projection" | "bf16-projection-f32")
}

/// The sub-category of every node of the plan (plus the block-progress slot), from its label and
/// its place.
fn classify(labels: &[NodeLabel]) -> Vec<usize> {
    let sub = |name: &str| SUBS.iter().position(|(s, _)| *s == name).unwrap();
    let mut out = Vec::with_capacity(labels.len() + 1);
    let (mut layer, mut mixer_seen, mut after_mixer) = (None, false, 0usize);
    for (i, l) in labels.iter().enumerate() {
        if l.layer != layer {
            layer = l.layer;
            mixer_seen = false;
            after_mixer = 0;
        }
        let op = l.operation;
        let s = if is_projection(op) {
            // The next node that is not the packing of activations.
            let next = labels[i + 1..].iter().map(|l| l.operation).find(|o| {
                !matches!(
                    *o,
                    "quantize-activations"
                        | "pack-activations"
                        | "bf16-projection"
                        | "bf16-projection-f32"
                )
            });
            if next.is_some_and(|o| o.starts_with("moe-route")) {
                sub("gemm-router")
            } else if !mixer_seen {
                sub("gemm-qkv")
            } else {
                after_mixer += 1;
                if after_mixer == 1 {
                    sub("gemm-o")
                } else {
                    sub("gemm-mlp")
                }
            }
        } else {
            if is_mixer(op) {
                mixer_seen = true;
            }
            match op {
                "attention" | "kv-append" | "kv-copy-k-to-v" => sub("attention"),
                "moe-gate-up" | "moe-down" => sub("gemm-experts"),
                "language-head-bf16" => sub("gemm-lm-head"),
                "quantize-activations"
                | "pack-activations"
                | "moe-quantize"
                | "quantize-head-activations" => sub("gemm-prep"),
                "short-convolution" => sub("recurrent"),
                o if o.starts_with("delta-") || o.starts_with("mamba-") => sub("recurrent"),
                "moe-route" | "moe-route-rows" | "moe-combine" | "moe-combine-rows"
                | "moe-qwen-out" | "moe-activation" => sub("moe-routing"),
                "block-begin" | "block-end" | "layer-end" | "block-progress" => sub("block"),
                "rms-norm"
                | "elementwise-binary"
                | "head-norm-rope"
                | "head-rope"
                | "value-head-norm"
                | "qkv-bias"
                | "head-gate"
                | "attention-output-gate"
                | "final-norm"
                | "logits-softcap"
                | "argmax"
                | "argmax-slices"
                | "token-embedding"
                | "embedding-scale"
                | "layer-scalar"
                | "layer-branch-gelu"
                | "dynamic-offset" => sub("elem"),
                o if o.starts_with("per-layer-") => sub("elem"),
                _ => sub("other"),
            }
        };
        out.push(s);
    }
    out.push(sub("block"));
    out
}

#[derive(Clone, Copy, Default)]
struct Acc {
    wall: f64,
    work: f64,
    wait: f64,
}

impl Acc {
    fn add(&mut self, o: Acc) {
        self.wall += o.wall;
        self.work += o.work;
        self.wait += o.wait;
    }
}

/// What the calls of one prefill add up to.
pub struct PrefillProfile {
    rows: usize,
    calls: usize,
    seconds: f64,
    participants: usize,
    plan_nodes: usize,
    subs: Vec<usize>,
    /// `[layer slot][sub-category]`: slot 0 the nodes before the first layer, slot `l + 1` layer
    /// `l`, the last slot the nodes after the last layer.
    by_layer: Vec<[Acc; SUBS.len()]>,
    /// By operation name.
    by_op: Vec<(&'static str, Acc, usize)>,
    /// Per participant: work and wait seconds.
    per_thread: Vec<(f64, f64)>,
    /// Per participant: work seconds by core (the auxiliary word of the `work_start` stamp:
    /// `IA32_TSC_AUX` on x86-64, the core number of `tpidr_el0` on Darwin), and the changes of
    /// core from one node to the next.
    per_thread_cores: Vec<Vec<(u32, f64)>>,
    per_thread_moves: Vec<usize>,
    /// The wall of every node split at its last arrival: `critical` from the last departure of the
    /// previous node to the last arrival at this barrier (the slowest participant's work, plus
    /// its late start), `tail` from that last arrival to the last departure (how late the last
    /// waiter sees the barrier open). Seconds, summed over nodes.
    critical: f64,
    tail: f64,
    /// Summed over nodes and participants: departure − last arrival (every waiter's release
    /// delay).
    release: f64,
    /// Per participant: nodes where it arrived last, nodes where it departed last, and its summed
    /// late start (its `work_start` minus the previous node's last departure, when positive).
    last_arrivals: Vec<usize>,
    last_departures: Vec<usize>,
    late_start: Vec<f64>,
    /// The rows, wall seconds, and involuntary / voluntary context switches of the process
    /// (`getrusage`) of every call, in order.
    call_walls: Vec<(usize, f64, i64, i64)>,
    /// The barriers whose tail exceeds 50 µs, by operation (count, summed tail), by last-departing
    /// thread, and how often that thread's core differed between its arrival and its
    /// departure.
    slow_by_op: Vec<(&'static str, usize, f64)>,
    slow_by_thread: Vec<usize>,
    slow_core_changed: usize,
    slow_cores: Vec<(u32, u32)>,
    /// Barriers seen (nodes with a stamp, over blocks).
    barriers: usize,
    /// The barriers whose tail exceeds 50 µs, and their summed tail.
    slow_tails: usize,
    slow_tail_sum: f64,
    /// Sum over nodes of `participants × wall`: what work + wait is compared with.
    team_seconds: f64,
}

impl PrefillProfile {
    pub fn new() -> Self {
        PrefillProfile {
            rows: 0,
            calls: 0,
            seconds: 0.0,
            participants: 0,
            plan_nodes: 0,
            subs: Vec::new(),
            by_layer: Vec::new(),
            by_op: Vec::new(),
            per_thread: Vec::new(),
            per_thread_cores: Vec::new(),
            per_thread_moves: Vec::new(),
            critical: 0.0,
            tail: 0.0,
            release: 0.0,
            last_arrivals: Vec::new(),
            last_departures: Vec::new(),
            late_start: Vec::new(),
            call_walls: Vec::new(),
            slow_by_op: Vec::new(),
            slow_by_thread: Vec::new(),
            slow_core_changed: 0,
            slow_cores: Vec::new(),
            barriers: 0,
            slow_tails: 0,
            slow_tail_sum: 0.0,
            team_seconds: 0.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.calls == 0
    }

    /// Fold the trace of one call of the prefill program: `rows` fed in `seconds` of wall time.
    pub fn fold(
        &mut self,
        labels: &[NodeLabel],
        trace: &SyncTraceSnapshot,
        rows: usize,
        seconds: f64,
        switches: (i64, i64),
    ) {
        let (p, per) = (trace.participants, trace.nodes_per_block);
        if p == 0 || per == 0 || trace.blocks == 0 {
            return;
        }
        if self.plan_nodes != labels.len() || self.subs.is_empty() {
            self.subs = classify(labels);
            self.plan_nodes = labels.len();
            let layers = labels
                .iter()
                .filter_map(|l| l.layer)
                .max()
                .map_or(0, |m| m + 1);
            self.by_layer = vec![[Acc::default(); SUBS.len()]; layers + 2];
        }
        if self.per_thread.len() < p {
            self.per_thread.resize(p, (0.0, 0.0));
            self.per_thread_cores.resize(p, Vec::new());
            self.per_thread_moves.resize(p, 0);
            self.last_arrivals.resize(p, 0);
            self.last_departures.resize(p, 0);
            self.late_start.resize(p, 0.0);
            self.slow_by_thread.resize(p, 0);
        }
        let mut last_core: Vec<Option<u32>> = vec![None; p];
        self.participants = self.participants.max(p);
        let n_layers = self.by_layer.len() - 2;
        let slot_of = |node: usize, seen_layer: bool| match labels.get(node).and_then(|l| l.layer) {
            Some(l) => l + 1,
            None if seen_layer => n_layers + 1,
            None => 0,
        };
        let rec = |b: usize, n: usize| &trace.records[(b * per + n) * p..(b * per + n + 1) * p];
        let nz_max = |it: &mut dyn Iterator<Item = u64>| it.filter(|&v| v != 0).max();
        // The span of the call in stamps.
        let first = trace
            .records
            .iter()
            .map(|r| r.work_start)
            .filter(|&v| v != 0)
            .min()
            .unwrap_or(0);
        let mut last = first;
        for r in &trace.records {
            last = last.max(r.barrier_end).max(r.work_end);
        }
        let tick = seconds / last.saturating_sub(first).max(1) as f64;
        let mut prev = first;
        for b in 0..trace.blocks {
            let mut seen_layer = false;
            for n in 0..per {
                let rs = rec(b, n);
                let end = nz_max(&mut rs.iter().map(|r| r.barrier_end))
                    .or_else(|| nz_max(&mut rs.iter().map(|r| r.work_end)));
                let Some(end) = end else { continue };
                let wall = end.saturating_sub(prev) as f64 * tick;
                // The wait decomposed around the last arrival.
                if let Some(arrive) = nz_max(&mut rs.iter().map(|r| r.barrier_start)) {
                    self.barriers += 1;
                    let critical = arrive.saturating_sub(prev) as f64 * tick;
                    let tail = end.saturating_sub(arrive) as f64 * tick;
                    self.critical += critical;
                    self.tail += tail;
                    if tail > 50e-6 {
                        self.slow_tails += 1;
                        self.slow_tail_sum += tail;
                        let name = labels.get(n).map_or("block-progress", |l| l.operation);
                        match self.slow_by_op.iter_mut().find(|(o, _, _)| *o == name) {
                            Some(e) => {
                                e.1 += 1;
                                e.2 += tail;
                            }
                            None => self.slow_by_op.push((name, 1, tail)),
                        }
                        if let Some((t, r)) = rs
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| r.barrier_end != 0)
                            .max_by_key(|(_, r)| r.barrier_end)
                        {
                            self.slow_by_thread[t] += 1;
                            if r.barrier_start_aux != r.barrier_end_aux {
                                self.slow_core_changed += 1;
                            }
                            if self.slow_cores.len() < 64 {
                                self.slow_cores
                                    .push((r.barrier_start_aux, r.barrier_end_aux));
                            }
                        }
                    }
                    for (t, r) in rs.iter().enumerate() {
                        if r.barrier_end != 0 {
                            self.release += r.barrier_end.saturating_sub(arrive) as f64 * tick;
                        }
                        if r.work_start != 0 && r.work_start > prev {
                            self.late_start[t] += (r.work_start - prev) as f64 * tick;
                        }
                    }
                    if let Some((t, _)) = rs
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.barrier_start != 0)
                        .max_by_key(|(_, r)| r.barrier_start)
                    {
                        self.last_arrivals[t] += 1;
                    }
                    if let Some((t, _)) = rs
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.barrier_end != 0)
                        .max_by_key(|(_, r)| r.barrier_end)
                    {
                        self.last_departures[t] += 1;
                    }
                }
                prev = prev.max(end);
                let mut work = 0.0;
                let mut wait = 0.0;
                for (t, r) in rs.iter().enumerate() {
                    let w = if r.work_start != 0 && r.work_end != 0 {
                        r.work_end.saturating_sub(r.work_start) as f64 * tick
                    } else {
                        0.0
                    };
                    let s = if r.barrier_start != 0 && r.barrier_end != 0 {
                        r.barrier_end.saturating_sub(r.barrier_start) as f64 * tick
                    } else {
                        0.0
                    };
                    work += w;
                    wait += s;
                    self.per_thread[t].0 += w;
                    self.per_thread[t].1 += s;
                    if r.work_start != 0 {
                        let core = r.work_start_aux;
                        match self.per_thread_cores[t]
                            .iter_mut()
                            .find(|(c, _)| *c == core)
                        {
                            Some(e) => e.1 += w,
                            None => self.per_thread_cores[t].push((core, w)),
                        }
                        if last_core[t].is_some_and(|c| c != core) {
                            self.per_thread_moves[t] += 1;
                        }
                        last_core[t] = Some(core);
                    }
                }
                self.team_seconds += wall * p as f64;
                let acc = Acc { wall, work, wait };
                if labels.get(n).is_some_and(|l| l.layer.is_some()) {
                    seen_layer = true;
                }
                let sub = self.subs.get(n).copied().unwrap_or(SUBS.len() - 1);
                let slot = slot_of(n, seen_layer);
                self.by_layer[slot][sub].add(acc);
                let name = labels.get(n).map_or("block-progress", |l| l.operation);
                match self.by_op.iter_mut().find(|(o, _, _)| *o == name) {
                    Some(e) => {
                        e.1.add(acc);
                        e.2 += 1;
                    }
                    None => self.by_op.push((name, acc, 1)),
                }
            }
        }
        self.call_walls
            .push((rows, seconds, switches.0, switches.1));
        self.rows += rows;
        self.calls += 1;
        self.seconds += seconds;
    }

    fn total(&self) -> [Acc; SUBS.len()] {
        let mut t = [Acc::default(); SUBS.len()];
        for l in &self.by_layer {
            for (a, b) in t.iter_mut().zip(l.iter()) {
                a.add(*b);
            }
        }
        t
    }

    /// The report on stderr and the JSON line; the profile is emptied.
    pub fn report(&mut self) {
        if self.calls == 0 {
            return;
        }
        let ms = |s: f64| s * 1e3;
        let total = self.total();
        let wall: f64 = total.iter().map(|a| a.wall).sum();
        let work: f64 = total.iter().map(|a| a.work).sum();
        let wait: f64 = total.iter().map(|a| a.wait).sum();
        let cat = |c: &str, t: &[Acc; SUBS.len()]| {
            let mut a = Acc::default();
            for (i, (_, k)) in SUBS.iter().enumerate() {
                if *k == c {
                    a.add(t[i]);
                }
            }
            a
        };
        let mut s = String::new();
        let _ = writeln!(
            s,
            "[prefill-profile] {} rows, {} calls, {:.1} ms ({:.1} rows/s), {} participants; work {:.1} thread-ms, barrier wait {:.1} thread-ms, \
             (work + wait) / (participants × wall) = {:.3}",
            self.rows,
            self.calls,
            ms(self.seconds),
            self.rows as f64 / self.seconds.max(1e-12),
            self.participants,
            ms(work),
            ms(wait),
            (work + wait) / self.team_seconds.max(1e-12)
        );
        let _ = write!(s, "[prefill-profile] wall by category:");
        for c in CATS {
            let a = cat(c, &total);
            let _ = write!(
                s,
                "  {c} {:.1} ms ({:.1} %, work/wall·P {:.2})",
                ms(a.wall),
                100.0 * a.wall / wall.max(1e-12),
                a.work / (a.wall * self.participants as f64).max(1e-12)
            );
        }
        let _ = writeln!(s);
        let _ = write!(s, "[prefill-profile] wall by sub-category:");
        for (i, (name, _)) in SUBS.iter().enumerate() {
            if total[i].wall > 0.0 {
                let _ = write!(
                    s,
                    "  {name} {:.1} ms ({:.1} %)",
                    ms(total[i].wall),
                    100.0 * total[i].wall / wall.max(1e-12)
                );
            }
        }
        let _ = writeln!(s);
        let mut ops = self.by_op.clone();
        ops.sort_by(|a, b| b.1.wall.total_cmp(&a.1.wall));
        let _ = write!(
            s,
            "[prefill-profile] by operation (wall, work, wait in ms; nodes):"
        );
        for (o, a, n) in &ops {
            let _ = write!(
                s,
                "  {o} {:.1}/{:.1}/{:.1} ({n})",
                ms(a.wall),
                ms(a.work),
                ms(a.wait)
            );
        }
        let _ = writeln!(s);
        let _ = write!(s, "[prefill-profile] per thread (work/wait ms):");
        for (t, (w, b)) in self.per_thread.iter().enumerate() {
            let _ = write!(s, "  {t}:{:.1}/{:.1}", ms(*w), ms(*b));
        }
        let _ = writeln!(s);
        // The spread of the work: the slowest participant against the median one.
        let mut works: Vec<f64> = self.per_thread.iter().map(|(w, _)| *w).collect();
        works.sort_by(|a, b| a.total_cmp(b));
        if let (Some(&max), Some(&min)) = (works.last(), works.first()) {
            let median = works[works.len() / 2];
            let slowest = self
                .per_thread
                .iter()
                .position(|(w, _)| *w == max)
                .unwrap_or(0);
            let _ = writeln!(s, "[prefill-profile] work spread: min {:.1} ms, median {:.1} ms, max {:.1} ms (thread {slowest}), max/median {:.3}", ms(min), ms(median), ms(max), max / median.max(1e-12));
        }
        let _ = writeln!(
            s,
            "[prefill-profile] wait decomposed: {} barriers; critical path {:.1} ms, tail {:.1} ms (last arrival → last departure; {} tails > 50 µs = {:.1} ms), release {:.1} thread-ms",
            self.barriers,
            ms(self.critical),
            ms(self.tail),
            self.slow_tails,
            ms(self.slow_tail_sum),
            ms(self.release)
        );
        let list = |v: &[usize]| {
            v.iter()
                .enumerate()
                .map(|(t, n)| format!("{t}:{n}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let _ = writeln!(
            s,
            "[prefill-profile] last to arrive, by thread: {}",
            list(&self.last_arrivals)
        );
        let _ = writeln!(
            s,
            "[prefill-profile] last to depart, by thread: {}",
            list(&self.last_departures)
        );
        let _ = writeln!(
            s,
            "[prefill-profile] late start, ms by thread: {}",
            self.late_start
                .iter()
                .enumerate()
                .map(|(t, v)| format!("{t}:{:.1}", ms(*v)))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(
            s,
            "[prefill-profile] calls (rows, ms): {}",
            self.call_walls
                .iter()
                .map(|(r, w, _, _)| format!("{r}/{:.1}", ms(*w)))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = writeln!(
            s,
            "[prefill-profile] context switches of the process per call (involuntary/voluntary): {}",
            self.call_walls.iter().map(|(_, _, i, v)| format!("{i}/{v}")).collect::<Vec<_>>().join(" ")
        );
        let mut by_op = self.slow_by_op.clone();
        by_op.sort_by(|a, b| b.2.total_cmp(&a.2));
        let _ = writeln!(
            s,
            "[prefill-profile] tails > 50 µs by operation (count, ms): {}",
            by_op
                .iter()
                .map(|(o, n, t)| format!("{o} {n} {:.1}", ms(*t)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            s,
            "[prefill-profile] tails > 50 µs by last-departing thread: {}",
            list(&self.slow_by_thread)
        );
        let _ = writeln!(
            s,
            "[prefill-profile] tails > 50 µs: core of the last departer changed between arrival and departure {} of {}; (arrival→departure) cores: {}",
            self.slow_core_changed,
            self.slow_tails,
            self.slow_cores.iter().map(|(a, b)| format!("{a}→{b}")).collect::<Vec<_>>().join(" ")
        );
        let _ = write!(
            s,
            "[prefill-profile] per thread, cores by work share (core:%; moves):"
        );
        for (t, cores) in self.per_thread_cores.iter().enumerate() {
            let tot: f64 = cores.iter().map(|(_, w)| *w).sum::<f64>().max(1e-12);
            let mut c = cores.clone();
            c.sort_by(|a, b| b.1.total_cmp(&a.1));
            let list: Vec<String> = c
                .iter()
                .take(4)
                .map(|(k, w)| format!("{k}:{:.0}", 100.0 * w / tot))
                .collect();
            let _ = write!(
                s,
                "  {t}[{}; {}]",
                list.join(","),
                self.per_thread_moves.get(t).copied().unwrap_or(0)
            );
        }
        let _ = writeln!(s);
        let head: Vec<&str> = SUBS.iter().map(|(n, _)| *n).collect();
        let _ = writeln!(
            s,
            "[prefill-profile] by layer, wall ms: layer | {} | total | wait/(P·wall)",
            head.join(" | ")
        );
        let n_layers = self.by_layer.len().saturating_sub(2);
        for (slot, l) in self.by_layer.iter().enumerate() {
            let lw: f64 = l.iter().map(|a| a.wall).sum();
            if lw == 0.0 {
                continue;
            }
            let lwait: f64 = l.iter().map(|a| a.wait).sum();
            let name = if slot == 0 {
                "input".to_string()
            } else if slot == n_layers + 1 {
                "head".to_string()
            } else {
                format!("L{:02}", slot - 1)
            };
            let cells: Vec<String> = l.iter().map(|a| format!("{:.2}", ms(a.wall))).collect();
            let _ = writeln!(
                s,
                "[prefill-profile] {name} | {} | {:.2} | {:.2}",
                cells.join(" | "),
                ms(lw),
                lwait / (lw * self.participants as f64).max(1e-12)
            );
        }
        eprint!("{s}");
        // The JSON line.
        let accj = |a: &Acc| serde_json::json!({"wall_ms": ms(a.wall), "work_ms": ms(a.work), "wait_ms": ms(a.wait)});
        let cats: serde_json::Map<String, serde_json::Value> = CATS
            .iter()
            .map(|c| (c.to_string(), accj(&cat(c, &total))))
            .collect();
        let subs: serde_json::Map<String, serde_json::Value> = SUBS
            .iter()
            .enumerate()
            .map(|(i, (n, _))| (n.to_string(), accj(&total[i])))
            .collect();
        let layers: Vec<serde_json::Value> = self
            .by_layer
            .iter()
            .enumerate()
            .map(|(slot, l)| {
                let name = if slot == 0 {
                    "input".to_string()
                } else if slot == n_layers + 1 {
                    "head".to_string()
                } else {
                    format!("{}", slot - 1)
                };
                let m: serde_json::Map<String, serde_json::Value> = SUBS
                    .iter()
                    .enumerate()
                    .map(|(i, (n, _))| (n.to_string(), accj(&l[i])))
                    .collect();
                serde_json::json!({"layer": name, "subs": m})
            })
            .collect();
        let opsj: Vec<serde_json::Value> = ops.iter().map(|(o, a, n)| serde_json::json!({"op": o, "nodes": n, "wall_ms": ms(a.wall), "work_ms": ms(a.work), "wait_ms": ms(a.wait)})).collect();
        let threads: Vec<serde_json::Value> = self
            .per_thread
            .iter()
            .enumerate()
            .map(|(t, (w, b))| {
                let cores: serde_json::Map<String, serde_json::Value> =
                    self.per_thread_cores.get(t).map_or(&[][..], |v| &v[..]).iter().map(|(c, w)| (c.to_string(), serde_json::json!(ms(*w)))).collect();
                serde_json::json!({"work_ms": ms(*w), "wait_ms": ms(*b), "cores_work_ms": cores, "moves": self.per_thread_moves.get(t).copied().unwrap_or(0)})
            })
            .collect();
        let j = serde_json::json!({
            "rows": self.rows, "calls": self.calls, "wall_ms": ms(self.seconds), "participants": self.participants,
            "work_ms": ms(work), "wait_ms": ms(wait), "team_ms": ms(self.team_seconds),
            "barriers": self.barriers, "critical_ms": ms(self.critical), "tail_ms": ms(self.tail), "release_ms": ms(self.release),
            "slow_tails": self.slow_tails, "slow_tail_ms": ms(self.slow_tail_sum),
            "last_arrivals": self.last_arrivals, "last_departures": self.last_departures,
            "late_start_ms": self.late_start.iter().map(|v| ms(*v)).collect::<Vec<_>>(),
            "calls_ms": self.call_walls.iter().map(|(r, w, i, v)| serde_json::json!([r, ms(*w), i, v])).collect::<Vec<_>>(),
            "slow_by_op": by_op.iter().map(|(o, n, t)| serde_json::json!([o, n, ms(*t)])).collect::<Vec<_>>(),
            "slow_by_thread": self.slow_by_thread, "slow_core_changed": self.slow_core_changed, "slow_cores": self.slow_cores,
            "categories": cats, "subcategories": subs, "operations": opsj, "threads": threads, "layers": layers,
        });
        eprintln!("[prefill-profile-json] {j}");
        *self = PrefillProfile::new();
    }
}

impl Default for PrefillProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// The involuntary and voluntary context switches of the process so far (`getrusage(RUSAGE_SELF)`;
/// zeros where it fails).
pub fn context_switches() -> (i64, i64) {
    #[cfg(unix)]
    {
        // SAFETY: `getrusage` fills the local struct.
        let mut u: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) } == 0 {
            return (u.ru_nivcsw as i64, u.ru_nvcsw as i64);
        }
    }
    (0, 0)
}

/// `HERBERT_PREFILL_PROFILE` set (to anything but `0` or empty).
pub fn requested() -> bool {
    std::env::var("HERBERT_PREFILL_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(op: &'static str, layer: Option<usize>) -> NodeLabel {
        NodeLabel {
            operation: op,
            layer,
        }
    }

    #[test]
    fn projections_are_told_apart_by_their_place() {
        let labels = [
            l("block-begin", None),
            l("rms-norm", Some(0)),
            l("quantize-activations", Some(0)),
            l("bf16-projection", Some(0)),
            l("bf16-projection", Some(0)),
            l("attention", Some(0)),
            l("bf16-projection", Some(0)),
            l("rms-norm", Some(0)),
            l("bf16-projection", Some(0)),
            l("moe-route-rows", Some(0)),
            l("moe-gate-up", Some(0)),
            l("bf16-projection", Some(0)),
            l("bf16-projection", Some(0)),
            l("mystery", Some(0)),
            l("language-head-bf16", None),
        ];
        let names: Vec<&str> = classify(&labels).iter().map(|&i| SUBS[i].0).collect();
        assert_eq!(
            names,
            [
                "block",
                "elem",
                "gemm-prep",
                "gemm-qkv",
                "gemm-qkv",
                "attention",
                "gemm-o",
                "elem",
                "gemm-router",
                "moe-routing",
                "gemm-experts",
                "gemm-mlp",
                "gemm-mlp",
                "other",
                "gemm-lm-head",
                "block"
            ]
        );
    }
}
