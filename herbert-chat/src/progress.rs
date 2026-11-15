//! How far a load has gone, for a server that answers while its model loads (`/props` and
//! `/health` of `herbert-serve`). The load names its stages and the share of the whole each takes;
//! a stage with a loop reports its steps. One load at a time per process: the state is global.

use std::sync::Mutex;
use std::time::Instant;

struct State {
    stage: &'static str,
    /// The stage's share of the whole: from `from` to `to`, `done / total` of the way.
    from: f32,
    to: f32,
    done: usize,
    total: usize,
    started: Option<Instant>,
}

static STATE: Mutex<State> = Mutex::new(State {
    stage: "",
    from: 0.0,
    to: 0.0,
    done: 0,
    total: 0,
    started: None,
});

/// What a load reports: its stage, the fraction of the whole (0 to 1) and the seconds since it
/// began.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub stage: &'static str,
    pub fraction: f32,
    pub seconds: f64,
}

/// A load begins.
pub fn begin() {
    let mut s = STATE.lock().unwrap();
    *s = State {
        stage: "start",
        from: 0.0,
        to: 0.0,
        done: 0,
        total: 0,
        started: Some(Instant::now()),
    };
}

/// A stage that takes the load from `from` to `to` (fractions of the whole).
pub fn stage(name: &'static str, from: f32, to: f32) {
    let mut s = STATE.lock().unwrap();
    (s.stage, s.from, s.to, s.done, s.total) = (name, from, to, 0, 0);
}

/// `done` of the stage's `total` steps.
pub fn step(done: usize, total: usize) {
    let mut s = STATE.lock().unwrap();
    (s.done, s.total) = (done.min(total), total);
}

/// The load is over.
pub fn finish() {
    let mut s = STATE.lock().unwrap();
    (s.stage, s.from, s.to, s.done, s.total) = ("ready", 1.0, 1.0, 0, 0);
}

pub fn snapshot() -> Snapshot {
    let s = STATE.lock().unwrap();
    let within = if s.total > 0 {
        s.done as f32 / s.total as f32
    } else {
        0.0
    };
    Snapshot {
        stage: s.stage,
        fraction: (s.from + (s.to - s.from) * within).clamp(0.0, 1.0),
        seconds: s.started.map_or(0.0, |t| t.elapsed().as_secs_f64()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stage_moves_the_whole_within_its_share() {
        begin();
        stage("q4", 0.25, 0.85);
        step(127, 254);
        let s = snapshot();
        assert_eq!(s.stage, "q4");
        assert!((s.fraction - 0.55).abs() < 1e-6);
        finish();
        assert_eq!(snapshot().fraction, 1.0);
    }
}
