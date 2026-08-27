//! Synthetic long-horizon evaluation scenarios with ground truth (Phase 5
//! of the design): deterministic event streams exercising the three
//! abilities LongMemEval shows frontier models fail — knowledge updates
//! (supersession), temporal projection, and multi-hop reasoning — plus
//! conflict abstention. `run_eval` measures answer accuracy, context token
//! cost vs. the raw transcript, and maintenance/query latency.

use crate::agent::{AgentMemory, MockExtractor};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::time::Instant;

/// Deterministic xorshift RNG so every run of a seed is identical.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub struct Scenario {
    rng: Rng,
    pub people: Vec<String>,
    pub orgs: Vec<String>,
    employer: HashMap<String, String>,
    managers: Vec<(String, String)>,
    likes: HashMap<String, BTreeSet<String>>,
    pub transcript_chars: usize,
    pub supersessions: usize,
}

impl Scenario {
    pub fn new(seed: u64, people: usize, orgs: usize) -> Self {
        Scenario {
            rng: Rng(seed | 1),
            people: (0..people).map(|i| format!("p{i}")).collect(),
            orgs: (0..orgs).map(|i| format!("org{i}")).collect(),
            employer: HashMap::new(),
            managers: Vec::new(),
            likes: HashMap::new(),
            transcript_chars: 0,
            supersessions: 0,
        }
    }

    /// Generate one turn's episode; ground truth is updated in lockstep.
    pub fn next_turn(&mut self, turn: usize) -> String {
        let ts = (turn * 10) as i64;
        let mut lines = Vec::new();
        match turn % 4 {
            0 => {
                // employment event: hire or change (knowledge update)
                let p = self.people[self.rng.below(self.people.len())].clone();
                let o = self.orgs[self.rng.below(self.orgs.len())].clone();
                if self.employer.contains_key(&p) {
                    self.supersessions += 1;
                }
                self.employer.insert(p.clone(), o.clone());
                lines.push(format!("{p} --works_at--> {o}"));
            }
            1 => {
                // manager link (multi-hop edges accumulate)
                let i = self.rng.below(self.people.len());
                let mut j = self.rng.below(self.people.len());
                if i == j {
                    j = (j + 1) % self.people.len();
                }
                let (a, b) = (self.people[i].clone(), self.people[j].clone());
                if !self.managers.contains(&(a.clone(), b.clone())) {
                    self.managers.push((a.clone(), b.clone()));
                }
                lines.push(format!("{a} --manager--> {b}"));
            }
            2 => {
                // non-exclusive preference: conflicts must stay open
                let p = self.people[self.rng.below(self.people.len())].clone();
                let o = self.orgs[self.rng.below(self.orgs.len())].clone();
                self.likes.entry(p.clone()).or_default().insert(o.clone());
                lines.push(format!("{p} --likes--> {o}"));
            }
            _ => {
                // filler small talk: extraction should skip it
                lines.push(format!("p{} checked in at t{ts}", self.rng.below(self.people.len())));
            }
        }
        let text = lines.join("\n");
        self.transcript_chars += text.len() + 1;
        text
    }

    /// (person, current org) questions for everyone ever employed.
    pub fn employment_questions(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self
            .employer
            .iter()
            .map(|(p, o)| (p.clone(), o.clone()))
            .collect();
        v.sort();
        v
    }

    fn reachable(&self, from: &str, to: &str) -> bool {
        let mut seen = BTreeSet::new();
        let mut q = VecDeque::from([from.to_string()]);
        while let Some(n) = q.pop_front() {
            if n == to {
                return true;
            }
            if !seen.insert(n.clone()) {
                continue;
            }
            for (a, b) in &self.managers {
                if a == &n {
                    q.push_back(b.clone());
                }
            }
        }
        false
    }

    /// Sampled ((from, to), expected-reachable) multi-hop questions mixing
    /// positives and negatives (balanced where the graph allows).
    pub fn multihop_questions(&mut self, n: usize) -> Vec<((String, String), bool)> {
        // oversample distinct pairs, then balance the classes afterwards —
        // a dense closure yields few negatives and must not starve the set
        let mut pos: Vec<((String, String), bool)> = Vec::new();
        let mut neg: Vec<((String, String), bool)> = Vec::new();
        let mut guard = 0;
        while pos.len() + neg.len() < n * 3 && guard < n * 60 {
            guard += 1;
            let a = self.people[self.rng.below(self.people.len())].clone();
            let b = self.people[self.rng.below(self.people.len())].clone();
            if a == b {
                continue;
            }
            let dup = pos.iter().chain(neg.iter()).any(|((x, y), _)| x == &a && y == &b);
            if dup {
                continue;
            }
            let r = self.reachable(&a, &b);
            if r {
                pos.push(((a, b), true));
            } else {
                neg.push(((a, b), false));
            }
        }
        // take up to half positives and half negatives, then fill from
        // whatever remains (dense closures have few negatives)
        let half = n.div_ceil(2);
        let mut out: Vec<((String, String), bool)> = Vec::new();
        let pos_keep = pos.len().min(half);
        out.extend(pos.drain(..pos_keep));
        let neg_keep = neg.len().min(n - out.len());
        out.extend(neg.drain(..neg_keep));
        if out.len() < n {
            out.extend(pos); // dense closure: fill with positives
        }
        out.truncate(n);
        out
    }

    /// People holding conflicting (non-exclusive) preferences that must ALL
    /// still be retrievable — the abstention check.
    pub fn conflicted_people(&self) -> Vec<(String, Vec<String>)> {
        self.likes
            .iter()
            .filter(|(_, s)| s.len() > 1)
            .map(|(p, s)| (p.clone(), s.iter().cloned().collect()))
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct EvalReport {
    pub turns: usize,
    pub employment_q: usize,
    pub employment_correct: usize,
    pub multihop_q: usize,
    pub multihop_correct: usize,
    pub abstain_people: usize,
    pub abstain_correct: usize,
    pub supersessions: usize,
    pub context_tokens: usize,
    pub transcript_tokens: usize,
    pub maintain_ms: f64,
    pub ask_deep_ms: f64,
}

impl EvalReport {
    pub fn token_savings(&self) -> f64 {
        if self.context_tokens == 0 {
            0.0
        } else {
            self.transcript_tokens as f64 / self.context_tokens as f64
        }
    }

    pub fn accuracy(&self) -> f64 {
        let total = self.employment_q + self.multihop_q + self.abstain_people;
        if total == 0 {
            0.0
        } else {
            (self.employment_correct + self.multihop_correct + self.abstain_correct) as f64
                / total as f64
        }
    }
}

/// Run a full synthetic evaluation: stream the scenario through an
/// `AgentMemory`, then score against ground truth.
pub fn run_eval(seed: u64, people: usize, orgs: usize, turns: usize, multihop_q: usize) -> EvalReport {
    let mut sc = Scenario::new(seed, people, orgs);
    let mut m = AgentMemory::new(
        MockExtractor::new(0.9),
        "reports_to(X,Y) :- current(X,\"manager\",Y).\n\
         trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    )
    .unwrap();

    let mut maintain_ms = 0.0f64;
    for turn in 0..turns {
        let text = sc.next_turn(turn);
        m.observe_at(&text, (turn * 10) as i64);
        let t0 = Instant::now();
        m.maintain((turn * 10) as i64);
        maintain_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }

    let mut rep = EvalReport {
        turns,
        supersessions: sc.supersessions,
        transcript_tokens: sc.transcript_chars / 4,
        maintain_ms,
        ..Default::default()
    };

    // knowledge updates: current employer must match the latest assertion
    for (p, org) in sc.employment_questions() {
        rep.employment_q += 1;
        let goal = format!("current(\"{p}\", \"works_at\", O)");
        let got = m.ask(&goal).unwrap_or_default();
        if got.len() == 1 && got[0] == format!("O={org}") {
            rep.employment_correct += 1;
        }
    }

    // multi-hop: demand-driven queries vs. truth reachability
    let t0 = Instant::now();
    for ((a, b), expect) in sc.multihop_questions(multihop_q) {
        rep.multihop_q += 1;
        let got = m
            .ask_deep(&format!("reports_to(\"{a}\", \"{b}\")"))
            .unwrap_or_default();
        let ans = !got.is_empty();
        if ans == expect {
            rep.multihop_correct += 1;
        }
    }
    rep.ask_deep_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // abstention: conflicted preferences must all remain open
    for (p, orgs_liked) in sc.conflicted_people() {
        rep.abstain_people += 1;
        let got = m
            .ask(&format!("current(\"{p}\", \"likes\", O)"))
            .unwrap_or_default();
        let got_set: BTreeSet<String> = got
            .iter()
            .filter_map(|r| r.strip_prefix("O=").map(|s| s.to_string()))
            .collect();
        let want: BTreeSet<String> = orgs_liked.into_iter().collect();
        if got_set == want {
            rep.abstain_correct += 1;
        }
    }

    // token cost: assembled context for a sample question vs full transcript
    let sample_people: Vec<&str> = sc.people.iter().take(5).map(|s| s.as_str()).collect();
    let ctx = m.context(&sample_people, 400);
    rep.context_tokens = ctx.len() / 4;
    rep
}
