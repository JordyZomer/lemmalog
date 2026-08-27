//! The evaluation core: fact store, semiring annotations, stratification,
//! seminaive fixpoint evaluation, and provenance tracking.

use crate::ast::{Clause, CmpOp, Lit};
use crate::intern::{Interner, Term, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

/// Semiring annotation carried by every fact:
/// confidence in [0,1] (product t-norm on joins) and provenance
/// (set of episode/source ids, union on joins).
#[derive(Debug, Clone, PartialEq)]
pub struct Ann {
    pub conf: f64,
    pub prov: BTreeSet<String>,
}

impl Default for Ann {
    fn default() -> Self {
        Ann::unit()
    }
}

impl Ann {
    pub fn base(conf: f64, prov: impl IntoIterator<Item: Into<String>>) -> Self {
        Ann {
            conf: conf.clamp(0.0, 1.0),
            prov: prov.into_iter().map(Into::into).collect(),
        }
    }

    pub fn unit() -> Self {
        Ann {
            conf: 1.0,
            prov: BTreeSet::new(),
        }
    }

    /// Semiring product: how annotations combine across a rule body.
    pub fn join(&self, other: &Ann) -> Ann {
        Ann {
            conf: (self.conf * other.conf).clamp(0.0, 1.0),
            prov: self.prov.union(&other.prov).cloned().collect(),
        }
    }
}

pub type Key = (String, Vec<Value>);

/// One event in the streaming change feed. Downstream projections (e.g. a
/// vector index or a UI) subscribe via `changes_since`: additions cover
/// every new fact, `Retracted` marks explicit EDB removals, and `Cleared`
/// marks a derived relation that was wholesale rebuilt (scoped/program
/// recompute) — the signal to re-sync that predicate from scratch.
#[derive(Debug, Clone, PartialEq)]
pub enum Change {
    Added(u64, Key),
    Retracted(u64, Key),
    Cleared(u64, String),
}

impl Change {
    pub fn epoch(&self) -> u64 {
        match self {
            Change::Added(e, _) | Change::Retracted(e, _) | Change::Cleared(e, _) => *e,
        }
    }
}

/// How one derived fact was produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Support {
    /// Asserted directly (EDB).
    Base,
    /// Derived by rule `rule` from body facts `body`.
    Rule { rule: String, body: Vec<Key> },
}

#[derive(Debug, Clone)]
pub struct StoredFact {
    pub ann: Ann,
    pub supports: Vec<Support>,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub key: Vec<Value>,
    pub fact: StoredFact,
}

/// A predicate's tuple set: rows in insertion order, with a key map for
/// dedup and lazy per-position secondary indexes whose lookups return row
/// ids (cheap) instead of cloned keys.
#[derive(Debug, Default, Clone)]
pub struct Relation {
    pub rows: Vec<Row>,
    by_key: HashMap<Vec<Value>, usize>,
    /// position -> value -> row ids
    idx: HashMap<usize, HashMap<Value, Vec<usize>>>,
    /// Facts added since the last completed `run()` (epoch delta).
    pub pending: BTreeSet<Vec<Value>>,
}

/// Provenance witnesses per fact are capped: `why()` needs one derivation,
/// not the (possibly exponentially many) paths; uncapped supports made
/// duplicate emissions quadratic in memory and scan time.
pub const SUPPORT_CAP: usize = 4;

impl Relation {
    /// Insert or merge. Returns true if the fact is new.
    fn insert(&mut self, args: Vec<Value>, f: StoredFact) -> bool {
        if let Some(&id) = self.by_key.get(&args) {
            let existing = &mut self.rows[id].fact;
            // lattice-merge annotations: max confidence, union provenance
            existing.ann.conf = existing.ann.conf.max(f.ann.conf);
            existing.ann.prov.extend(f.ann.prov);
            if existing.supports.len() < SUPPORT_CAP {
                for s in f.supports {
                    if existing.supports.len() >= SUPPORT_CAP {
                        break;
                    }
                    if !existing.supports.contains(&s) {
                        existing.supports.push(s);
                    }
                }
            }
            false
        } else {
            let id = self.rows.len();
            self.index_row(id, &args);
            self.by_key.insert(args.clone(), id);
            self.rows.push(Row { key: args, fact: f });
            let key = self.rows[id].key.clone();
            self.pending.insert(key);
            true
        }
    }

    fn remove(&mut self, args: &[Value]) -> bool {
        let Some(&id) = self.by_key.get(args) else {
            return false;
        };
        let last = self.rows.len() - 1;
        self.unindex_row(id);
        self.by_key.remove(args);
        if id != last {
            // swap_remove moves the LAST row into the hole; its key map
            // and index entries must follow it. (swap_remove itself
            // returns the removed row, not the relocated one.)
            let last_key = self.rows[last].key.clone();
            self.unindex_row(last);
            self.rows.swap_remove(id); // last row is now at `id`
            self.by_key.insert(last_key, id);
            let relocated_key = self.rows[id].key.clone();
            self.index_row(id, &relocated_key);
        } else {
            self.rows.pop();
        }
        true
    }

    fn index_row(&mut self, id: usize, key: &[Value]) {
        for (pos, v) in key.iter().enumerate() {
            self.idx
                .entry(pos)
                .or_default()
                .entry(*v)
                .or_default()
                .push(id);
        }
    }

    fn unindex_row(&mut self, id: usize) {
        let key = self.rows[id].key.clone();
        for (pos, v) in key.iter().enumerate() {
            if let Some(bucket) = self.idx.get_mut(&pos).and_then(|m| m.get_mut(v)) {
                bucket.retain(|&i| i != id);
            }
        }
    }

    /// Clear all contents (scoped recompute).
    fn clear(&mut self) {
        self.rows.clear();
        self.by_key.clear();
        self.idx.clear();
        self.pending.clear();
    }

    /// Candidate row ids for a pattern given pre-resolved bound positions.
    /// Uses the bound position with the smallest bucket; falls back to a
    /// full scan when nothing is bound. Indexes are maintained eagerly on
    /// insert/remove, so this needs only `&self`.
    pub fn lookup(&self, bound: &[Option<Value>]) -> Vec<usize> {
        let mut best: Option<(usize, usize)> = None; // (size, pos)
        for (pos, b) in bound.iter().enumerate() {
            if let Some(v) = b {
                let size = self
                    .idx
                    .get(&pos)
                    .and_then(|m| m.get(v))
                    .map(|s| s.len())
                    .unwrap_or(0);
                if size == 0 {
                    return Vec::new(); // bound to a value with no facts
                }
                if best.map(|(s, _)| size < s).unwrap_or(true) {
                    best = Some((size, pos));
                }
            }
        }
        match best {
            Some((_, pos)) => {
                let v = bound[pos].unwrap();
                self.idx[&pos][&v]
                    .iter()
                    .copied()
                    .filter(|&id| {
                        bound
                            .iter()
                            .zip(&self.rows[id].key)
                            .all(|(b, v)| b.as_ref().map(|bv| bv == v).unwrap_or(true))
                    })
                    .collect()
            }
            None => (0..self.rows.len()).collect(),
        }
    }

    /// If the fact exists, merge annotations and record the support while
    /// under the witness cap; returns true (existing) either way.
    pub fn merge_ann(&mut self, args: &[Value], ann: &Ann, support: Support) -> bool {
        if let Some(&id) = self.by_key.get(args) {
            let existing = &mut self.rows[id].fact;
            existing.ann.conf = existing.ann.conf.max(ann.conf);
            existing.ann.prov.extend(ann.prov.iter().cloned());
            if existing.supports.len() < SUPPORT_CAP
                && !existing.supports.contains(&support)
            {
                existing.supports.push(support);
            }
            true
        } else {
            false
        }
    }

    pub fn get(&self, args: &[Value]) -> Option<&StoredFact> {
        self.by_key.get(args).map(|&id| &self.rows[id].fact)
    }

    pub fn contains(&self, args: &[Value]) -> bool {
        self.by_key.contains_key(args)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }
}


// ------------------------------------------------------------- environment

#[derive(Debug, Clone, Default)]
/// Backtracking environment with an undo trail (WAM-style): bindings are
/// recorded so a join can undo them on backtrack instead of cloning the
/// whole environment per candidate.
struct Env {
    map: HashMap<String, Value>,
    ann: Ann,
    body_keys: Vec<Key>,
    trail: Vec<(String, Option<Value>)>,
}

/// A backtrack point: trail length, body-key length, annotation snapshot.
#[derive(Clone)]
struct Mark {
    trail: usize,
    body: usize,
    ann: Ann,
}

impl Env {
    fn new() -> Self {
        Env {
            map: HashMap::new(),
            ann: Ann::unit(),
            body_keys: Vec::new(),
            trail: Vec::new(),
        }
    }

    fn bind(&mut self, v: &str, val: Value) {
        self.trail.push((v.to_string(), self.map.get(v).copied()));
        self.map.insert(v.to_string(), val);
    }

    fn mark(&self) -> Mark {
        Mark {
            trail: self.trail.len(),
            body: self.body_keys.len(),
            ann: self.ann.clone(),
        }
    }

    fn undo(&mut self, m: &Mark) {
        while self.trail.len() > m.trail {
            let (v, old) = self.trail.pop().unwrap();
            match old {
                Some(prev) => {
                    self.map.insert(v, prev);
                }
                None => {
                    self.map.remove(&v);
                }
            }
        }
        self.body_keys.truncate(m.body);
        self.ann = m.ann.clone();
    }

    fn lookup(&self, t: &Term) -> Option<Value> {
        match t {
            Term::Var(v) => self.map.get(v).copied(),
            _ => None,
        }
    }
}

pub struct Engine {
    pub interner: Interner,
    pub relations: HashMap<String, Relation>,
    pub clauses: Vec<Clause>,
    pub now: i64,
    /// Facts derived by the most recent ask_deep (demand-slice size).
    pub last_demand_facts: usize,
    /// Facts the most recent `hypothetical` would have added.
    pub last_hypothetical_facts: usize,
    /// Append-only log of newly created facts: (epoch, key). Backs
    /// `changes_from` for context assembly ("what changed in memory").
    pub change_log: Vec<(u64, Key)>,
    /// Full change feed (adds, retractions, clears) for streaming
    /// projections.
    pub feed: Vec<Change>,
    /// Rule registry: (batch id, source, clause range end).
    pub rule_batches: Vec<(String, String, usize)>,
    /// Set when the program changed since the last run; the next `run()`
    /// clears and rebuilds every derived relation (backfill).
    pub program_dirty: bool,
    /// Predicates that have EVER had a defining rule: an over-approximation
    /// used to clear orphaned derivations after `uninstall`.
    pub(crate) ever_derived: BTreeSet<String>,
    /// change_log length at the end of the last completed `run()`: the
    /// additions-since window used to detect growth of negated predicates.
    last_run_log_len: usize,
    epoch: u64,
    /// Predicates retracted since the last `run()`; triggers a scoped
    /// recompute of their transitive dependents.
    retracted: BTreeSet<String>,
}

#[derive(Debug)]
pub struct StratError(pub String);
impl std::fmt::Display for StratError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stratification error: {}", self.0)
    }
}
impl std::error::Error for StratError {}

impl Engine {
    pub fn new() -> Self {
        Engine {
            interner: Interner::new(),
            relations: HashMap::new(),
            clauses: Vec::new(),
            now: 0,
            last_demand_facts: 0,
            last_hypothetical_facts: 0,
            change_log: Vec::new(),
            feed: Vec::new(),
            rule_batches: Vec::new(),
            program_dirty: true,
            ever_derived: BTreeSet::new(),
            last_run_log_len: 0,
            epoch: 0,
            retracted: BTreeSet::new(),
        }
    }

    // ------------------------------------------------------------- EDB API

    /// Assert a base fact with confidence and provenance.
    pub fn declare(&mut self, pred: &str, args: &[Value], ann: Ann) -> bool {
        let rel = self.relations.entry(pred.to_string()).or_default();
        let is_new = rel.insert(
            args.to_vec(),
            StoredFact {
                ann,
                supports: vec![Support::Base],
            },
        );
        if is_new {
            let key = (pred.to_string(), args.to_vec());
            self.feed.push(Change::Added(self.epoch, key.clone()));
            self.change_log.push((self.epoch, key));
        }
        is_new
    }

    /// Hypothetical evaluation: assert `extra` base facts, run to
    /// fixpoint, answer `goal`, then restore the store byte-identically —
    /// the "what follows if we assume X?" lookahead primitive (design
    /// §4.5). `last_hypothetical_facts` reports how many facts the
    /// assumption would have added.
    pub fn hypothetical(
        &mut self,
        extra: &[(&str, &[Value])],
        goal: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // validate the goal before touching anything
        let prog = crate::ast::parse_program(&format!("{goal}."))?;
        if prog.len() != 1 || !prog[0].is_fact {
            return Err(crate::ast::ParseError(format!(
                "hypothetical: expected a single atom, got {goal:?}"
            ))
            .into());
        }
        let backup_relations = self.relations.clone();
        let backup_log = self.change_log.clone();
        let backup_feed = self.feed.clone();
        let backup_retracted = self.retracted.clone();
        let backup_dirty = self.program_dirty;
        let backup_log_len = self.last_run_log_len;
        let backup_epoch = self.epoch;
        let log_mark = self.change_log.len();

        for (pred, args) in extra {
            self.declare(pred, args, Ann::unit());
        }
        self.run();
        self.last_hypothetical_facts = self.change_log.len() - log_mark;
        let rows = self.ask(goal)?;

        self.relations = backup_relations;
        self.change_log = backup_log;
        self.feed = backup_feed;
        self.retracted = backup_retracted;
        self.program_dirty = backup_dirty;
        self.last_run_log_len = backup_log_len;
        self.epoch = backup_epoch;
        Ok(rows)
    }

    /// Facts created in the given epoch or later (inclusive): the window a
    /// turn's assertions and derivations are logged under, since the epoch
    /// counter only advances at the end of `run()`.
    pub fn changes_from(&self, epoch: u64) -> Vec<Key> {
        self.change_log
            .iter()
            .filter(|(e, _)| *e >= epoch)
            .map(|(_, k)| k.clone())
            .collect()
    }

    pub fn sym(&mut self, s: &str) -> Value {
        Value::Sym(self.interner.intern(s))
    }

    /// Retract a base (EDB) fact. Used for supersession when the old tuple
    /// must be replaced (e.g. re-asserting an edge with a closed `valid_to`).
    /// The next `run()` recomputes only the transitive dependents of the
    /// retracted predicate (scoped negative delta); unrelated derived
    /// relations keep their incrementality.
    pub fn retract(&mut self, pred: &str, args: &[Value]) -> bool {
        if let Some(rel) = self.relations.get_mut(pred) {
            let existed = rel.remove(args);
            rel.pending.remove(args);
            if existed {
                self.feed
                    .push(Change::Retracted(self.epoch, (pred.to_string(), args.to_vec())));
                self.retracted.insert(pred.to_string());
            }
            existed
        } else {
            false
        }
    }

    /// The streaming change feed: all events logged at or after the given
    /// checkpoint epoch. Events asserted between runs are stamped with the
    /// current epoch (the counter only advances at the end of `run()`), so
    /// checkpoint at `epoch()` right after a run and everything from the
    /// next turn's window arrives — additions, retractions, and wholesale
    /// clears — exactly what an external projection needs to stay in sync.
    pub fn changes_since(&self, epoch: u64) -> Vec<Change> {
        self.feed
            .iter()
            .filter(|c| c.epoch() >= epoch)
            .cloned()
            .collect()
    }

    pub fn set_now(&mut self, now: i64) {
        self.now = now;
    }

    // ----------------------------------------------------------- stratify

    /// Stratum assignment: depth(p) = max over rules defining p of
    ///   depth(q)        for positive dependencies, and
    ///   depth(q) + 1    for negated dependencies.
    /// Rejects programs where a negative edge lies on a dependency cycle.
    pub fn strata(&self) -> Result<Vec<Vec<usize>>, StratError> {
        let mut defines: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, c) in self.clauses.iter().enumerate() {
            if !c.is_fact {
                defines.entry(c.head.pred.as_str()).or_default().push(i);
            }
        }
        // dependency graph over IDB predicates (head -> body dep)
        let mut deps: HashMap<&str, Vec<(&str, bool)>> = HashMap::new();
        for c in &self.clauses {
            if c.is_fact {
                continue;
            }
            for lit in &c.body {
                let (neg, dep) = match lit {
                    Lit::Neg(a) => (true, a.pred.as_str()),
                    Lit::Pos(a) => (false, a.pred.as_str()),
                    Lit::Cmp(..) | Lit::Now(_) => continue,
                };
                if defines.contains_key(dep) {
                    deps.entry(c.head.pred.as_str())
                        .or_default()
                        .push((dep, neg));
                }
            }
        }
        // negation-cycle check: for every negated edge h -> b, b must not
        // reach h through the dependency graph
        for (h, edges) in &deps {
            for (b, neg) in edges {
                if *neg && reaches(&deps, b, h) {
                    return Err(StratError(format!(
                        "unstratifiable: negation of {b} in a dependency cycle with {h}"
                    )));
                }
            }
        }
        // aggregates must not be read by their own bodies (recursively):
        // folding requires the body to be complete first
        let agg_heads: BTreeSet<&str> = self
            .clauses
            .iter()
            .filter(|c| !c.is_fact && Self::is_agg_clause(c))
            .map(|c| c.head.pred.as_str())
            .collect();
        for c in &self.clauses {
            if c.is_fact || !Self::is_agg_clause(c) {
                continue;
            }
            for lit in &c.body {
                if let Lit::Pos(a) | Lit::Neg(a) = lit {
                    if a.pred == c.head.pred || reaches(&deps, &a.pred, &c.head.pred) {
                        return Err(StratError(format!(
                            "aggregation over recursive dependency: {} reads {}",
                            c.head.pred, a.pred
                        )));
                    }
                }
            }
        }
        // mixed definition: a predicate defined by aggregation clauses may
        // not also have ordinary defining clauses (fold/merge semantics
        // would collide)
        for c in &self.clauses {
            if c.is_fact {
                continue;
            }
            if !Self::is_agg_clause(c) && agg_heads.contains(c.head.pred.as_str()) {
                return Err(StratError(format!(
                    "mixed definition: {} is aggregated in one clause and ordinary in another",
                    c.head.pred
                )));
            }
        }
        // stratification by SCC condensation: predicates in the same
        // strongly-connected component (mutual/self recursion) share a
        // stratum; every cross-SCC dependency is strictly ordered, so a
        // scoped recompute can clear and rebuild level by level with
        // same-pass reader propagation (no extra full-rebuild rounds)
        let idb_list: Vec<&str> = defines.keys().copied().collect();
        let index_of: HashMap<&str, usize> =
            idb_list.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let n = idb_list.len();
        // Tarjan SCC over head -> dep edges
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (h, edges) in &deps {
            if let Some(&hi) = index_of.get(*h) {
                for (d, _) in edges {
                    if let Some(&di) = index_of.get(*d) {
                        adj[hi].push(di);
                    }
                }
            }
        }
        let mut scc_of: Vec<usize> = vec![usize::MAX; n];
        let mut low = vec![0usize; n];
        let mut disc = vec![0usize; n];
        let mut on_stack = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut next_disc = 1usize;
        let mut scc_count = 0usize;
        for start_node in 0..n {
            if disc[start_node] != 0 {
                continue;
            }
            // iterative Tarjan
            let mut call: Vec<(usize, usize)> = vec![(start_node, 0)];
            while let Some((v, ai)) = call.pop() {
                if ai == 0 {
                    disc[v] = next_disc;
                    low[v] = next_disc;
                    next_disc += 1;
                    stack.push(v);
                    on_stack[v] = true;
                }
                let mut recursed = false;
                for i in ai..adj[v].len() {
                    let w = adj[v][i];
                    if disc[w] == 0 {
                        call.push((v, i + 1));
                        call.push((w, 0));
                        recursed = true;
                        break;
                    } else if on_stack[w] {
                        low[v] = low[v].min(disc[w]);
                    }
                }
                if recursed {
                    continue;
                }
                if low[v] == disc[v] {
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        scc_of[w] = scc_count;
                        if w == v {
                            break;
                        }
                    }
                    scc_count += 1;
                }
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
            }
        }
        // condensation: longest-path level per SCC (Tarjan pops SCCs in
        // reverse topological order w.r.t. head->dep edges, i.e. dependents
        // first; process in pop order to propagate levels)
        let mut scc_level = vec![0u32; scc_count];
        for v in 0..n {
            // process in Tarjan pop order: scc ids increase along pops
            let _ = v;
        }
        // edges between SCCs: level[scc_head] >= level[scc_dep] + 1
        // iterate to fixpoint over condensation edges (small graphs)
        let mut cond_edges: Vec<(usize, usize)> = Vec::new();
        for (h, edges) in &deps {
            if let Some(&hi) = index_of.get(*h) {
                for (d, _) in edges {
                    if let Some(&di) = index_of.get(*d) {
                        if scc_of[hi] != scc_of[di] {
                            cond_edges.push((scc_of[hi], scc_of[di]));
                        }
                    }
                }
            }
        }
        loop {
            let mut changed = false;
            for (hs, ds) in &cond_edges {
                if scc_level[*hs] < scc_level[*ds] + 1 {
                    scc_level[*hs] = scc_level[*ds] + 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut depth: HashMap<String, u32> = HashMap::new();
        for (i, p) in idb_list.iter().enumerate() {
            depth.insert(p.to_string(), scc_level[scc_of[i]]);
        }

        let mut strata: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, c) in self.clauses.iter().enumerate() {
            if c.is_fact {
                continue;
            }
            let d = depth[&c.head.pred];
            strata.entry(d).or_default().push(i);
        }
        Ok(strata.into_values().collect())
    }

    // -------------------------------------------------------------- fixpoint

    /// Run (or incrementally re-run) the program. Returns the number of new
    /// facts derived this epoch. Pending deltas are consumed: a subsequent
    /// `run()` with no new assertions derives nothing.
    pub fn run(&mut self) -> usize {
        let strata_pre = self.strata().ok();
        if self.program_dirty {
            if let Some(s) = &strata_pre {
                self.program_recompute(s);
            }
        }
        // program-embedded facts
        let fact_clauses: Vec<Clause> = self
            .clauses
            .iter()
            .filter(|c| c.is_fact)
            .cloned()
            .collect();
        for c in &fact_clauses {
            if let Some(args) = self.ground_args(&c.head.args) {
                self.declare(&c.head.pred, &args, Ann::unit());
            }
        }

        let strata = match self.strata() {
            Ok(s) => s,
            Err(_) => return 0, // use check_program to surface the error
        };
        // main evaluation: strata in order (negated predicates are always
        // lower-stratum or EDB, so within-run negation reads are complete)
        let mut derived = 0;
        for stratum in &strata {
            derived += self.eval_stratum(stratum);
        }
        // invalidation passes AFTER evaluation: retractions (supersession)
        // and cross-run growth of negated predicates leave stale derived
        // rows; clear and rebuild the affected slices now that all inputs
        // are materialized. Loop to a fixpoint (aggregate value changes
        // mark their heads retracted and cascade here too).
        let neg_rules = self
            .clauses
            .iter()
            .any(|c| !c.is_fact && c.body.iter().any(|l| matches!(l, Lit::Neg(_))));
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > strata.len() + 2 {
                break;
            }
            let mut extra_clear: BTreeSet<String> = BTreeSet::new();
            if neg_rules {
                let neg_readers: BTreeSet<(String, String)> = self
                    .clauses
                    .iter()
                    .filter(|c| !c.is_fact)
                    .flat_map(|c| {
                        c.body.iter().filter_map(move |l| match l {
                            Lit::Neg(a) => Some((a.pred.clone(), c.head.pred.clone())),
                            _ => None,
                        })
                    })
                    .collect();
                let grew: BTreeSet<&String> = self.change_log[self.last_run_log_len..]
                    .iter()
                    .map(|(_, (p, _))| p)
                    .collect();
                for (neg_pred, reader) in &neg_readers {
                    if grew.contains(neg_pred) {
                        extra_clear.insert(reader.clone());
                    }
                }
            }
            if self.retracted.is_empty() && extra_clear.is_empty() {
                break;
            }
            let (_, d) = self.scoped_recompute(&strata, extra_clear);
            derived += d;
        }
        self.last_run_log_len = self.change_log.len();
        for rel in self.relations.values_mut() {
            rel.pending.clear();
        }
        self.last_run_log_len = self.change_log.len();
        self.epoch += 1;
        derived
    }

    /// Scoped negative delta: after retractions, clear and re-derive only the
    /// derived predicates that transitively read a retracted predicate.
    /// Unaffected derived relations are left intact; seeds are the full
    /// contents of the inputs the cleared predicates read from.
    /// Scoped negative delta, DRed-style: process dependent predicates
    /// level by level (stratum order). Clear and rebuild a dependent only
    /// if the key set of what it reads actually changed — a works_at
    /// supersession rebuilds `current` (linear) but leaves a closure that
    /// only reads the manager slice untouched. Propagation continues to
    /// deeper dependents only through predicates whose contents changed.
    fn scoped_recompute(
        &mut self,
        strata: &[Vec<usize>],
        extra_clear: BTreeSet<String>,
    ) -> (BTreeSet<usize>, usize) {
        // direct readers: pred -> IDB preds whose rules read it (pos/neg).
        // Aggregation clauses read via their temp relation and fold:
        // body -> temp -> head.
        let mut direct_readers: std::collections::BTreeMap<String, BTreeSet<String>> =
            Default::default();
        for (ci, c) in self.clauses.iter().enumerate() {
            if c.is_fact {
                continue;
            }
            if Self::is_agg_clause(c) {
                let temp = self.agg_temp_pred(ci);
                for lit in &c.body {
                    if let Lit::Pos(a) | Lit::Neg(a) = lit {
                        direct_readers
                            .entry(a.pred.clone())
                            .or_default()
                            .insert(temp.clone());
                    }
                }
                direct_readers
                    .entry(temp)
                    .or_default()
                    .insert(c.head.pred.clone());
            } else {
                for lit in &c.body {
                    if let Lit::Pos(a) | Lit::Neg(a) = lit {
                        direct_readers
                            .entry(a.pred.clone())
                            .or_default()
                            .insert(c.head.pred.clone());
                    }
                }
            }
        }
        let mut idb: BTreeSet<String> = strata
            .iter()
            .flat_map(|s| s.iter().map(|&i| self.clauses[i].head.pred.clone()))
            .collect();
        // aggregation temp relations are derived state too (they are not
        // clause heads in strata, so add them explicitly)
        for ci in 0..self.clauses.len() {
            if !self.clauses[ci].is_fact && Self::is_agg_clause(&self.clauses[ci]) {
                idb.insert(self.agg_temp_pred(ci));
            }
        }
        // pending-clear worklist starts at direct IDB dependents of
        // retracted predicates, plus any explicitly requested clears
        let mut to_clear: BTreeSet<String> = self
            .retracted
            .iter()
            .chain(extra_clear.iter())
            .flat_map(|p| direct_readers.get(p).cloned().unwrap_or_default())
            .filter(|p| idb.contains(p))
            .collect();
        for p in extra_clear {
            if idb.contains(&p) {
                to_clear.insert(p);
            }
        }
        self.retracted.clear();
        let mut evaluated: BTreeSet<usize> = BTreeSet::new();
        let mut derived = 0usize;
        // fixpoint over rounds: a change can queue readers in an
        // already-processed stratum (same-depth positive dependency), which
        // demands another pass
        let mut round = 0usize;
        let mut preexisting: std::collections::BTreeMap<String, BTreeSet<Vec<Value>>> =
            Default::default();
        while !to_clear.is_empty() && round < strata.len() + 2 {
            round += 1;
            for (si, stratum) in strata.iter().enumerate() {
                if to_clear.is_empty() {
                    break;
                }
            let mut here: BTreeSet<String> = stratum
                .iter()
                .map(|&i| self.clauses[i].head.pred.clone())
                .filter(|p| to_clear.contains(p))
                .collect();
            // aggregation temps attach to their clause's stratum
            for &i in stratum {
                if !self.clauses[i].is_fact && Self::is_agg_clause(&self.clauses[i]) {
                    let t = self.agg_temp_pred(i);
                    if to_clear.contains(&t) {
                        here.insert(t);
                    }
                }
            }
            if here.is_empty() {
                continue;
            }
            to_clear = to_clear.difference(&here).cloned().collect();
            // snapshots to detect actual change
            let mut snapshots: std::collections::BTreeMap<String, BTreeSet<Vec<Value>>> =
                Default::default();
            for p in &here {
                if let Some(rel) = self.relations.get(p) {
                    snapshots.insert(
                        p.clone(),
                        rel.rows.iter().map(|r| r.key.clone()).collect(),
                    );
                }
            }
            for p in &here {
                if let Some(rel) = self.relations.get_mut(p) {
                    preexisting
                        .entry(p.clone())
                        .or_insert_with(|| rel.rows.iter().map(|r| r.key.clone()).collect());
                    if rel.len() > 0 {
                        self.feed.push(Change::Cleared(self.epoch, p.clone()));
                    }
                    rel.clear();
                }
            }
            // seed: full contents of the body predicates of these rules
            // that were not just cleared
            let mut seeds: BTreeSet<String> = BTreeSet::new();
            for (ci, c) in self.clauses.iter().enumerate() {
                if c.is_fact {
                    continue;
                }
                let owner = if Self::is_agg_clause(c) {
                    self.agg_temp_pred(ci)
                } else {
                    c.head.pred.clone()
                };
                if !here.contains(&owner) {
                    continue;
                }
                for lit in &c.body {
                    if let Lit::Pos(a) | Lit::Neg(a) = lit {
                        if !here.contains(&a.pred) {
                            seeds.insert(a.pred.clone());
                        }
                    }
                }
            }
            for p in seeds {
                if let Some(rel) = self.relations.get_mut(&p) {
                    let keys: Vec<Vec<Value>> = rel.rows.iter().map(|r| r.key.clone()).collect();
                    rel.pending.extend(keys);
                }
            }
            derived += self.eval_stratum(stratum);
            evaluated.insert(si);
            // changed predicates propagate clearing to their readers
            for p in &here {
                let now_keys: BTreeSet<Vec<Value>> = self
                    .relations
                    .get(p)
                    .map(|r| r.rows.iter().map(|x| x.key.clone()).collect())
                    .unwrap_or_default();
                let changed = now_keys != snapshots.get(p).cloned().unwrap_or_default();
                if changed {
                    if let Some(readers) = direct_readers.get(p) {
                        for r in readers {
                            // eval_stratum already fixpoints within a
                            // stratum (self-recursion and same-SCC readers
                            // see the final state); requeueing them only
                            // buys a redundant full-rebuild round
                            if r != p && !here.contains(r) && idb.contains(r) {
                                to_clear.insert(r.clone());
                            }
                        }
                    }
                }
            }
            }
        }
        // multi-round re-derivations of facts that existed before the
        // recompute are not new facts
        for (p, keys) in &preexisting {
            if let Some(rel) = self.relations.get(p) {
                let still: usize = keys
                    .iter()
                    .filter(|k| rel.contains(k))
                    .count();
                derived = derived.saturating_sub(still);
            }
        }
        (evaluated, derived)
    }

    /// Does this clause aggregate (has Agg terms in its head)?
    fn is_agg_clause(c: &Clause) -> bool {
        c.head.args.iter().any(|t| matches!(t, Term::Agg(..)))
    }

    /// Stable temp predicate for an aggregation clause's lowered body
    /// solutions: `__agg:{head_pred}:{clause_index}`.
    fn agg_temp_pred(&self, ci: usize) -> String {
        format!("__agg:{}:{ci}", self.clauses[ci].head.pred)
    }

    /// Lower an aggregation clause: head = temp(group args..., inner
    /// terms...); body unchanged. The ordinary evaluator enumerates all
    /// body solutions into the temp relation (set semantics = distinct
    /// (group, value) pairs, which is exactly COUNT(DISTINCT) and is
    /// idempotent for min/max/sum).
    fn lower_agg_clause(&self, ci: usize) -> Clause {
        let c = &self.clauses[ci];
        let mut args = Vec::new();
        for t in &c.head.args {
            match t {
                Term::Agg(_, inner) => args.push((**inner).clone()),
                other => args.push(other.clone()),
            }
        }
        Clause {
            name: c.name.clone(),
            head: crate::ast::Atom {
                pred: self.agg_temp_pred(ci),
                args,
            },
            body: c.body.clone(),
            is_fact: false,
        }
    }

    /// Fold one aggregation clause's temp relation into its head
    /// predicate. Groups are the temp relation's non-agg prefix; each agg
    /// position folds over its suffix column. Changed values replace the
    /// old rows (key change) so downstream recomputes propagate.
    fn fold_agg_clause(&mut self, ci: usize) -> usize {
        let c = self.clauses[ci].clone();
        let temp = self.agg_temp_pred(ci);
        let Some(rel) = self.relations.get(&temp) else {
            return 0;
        };
        // collect group data under this borrow; the fold writes below
        // need the borrow released
        let groups = {
            let group_len = c
                .head
                .args
                .iter()
                .filter(|t| !matches!(t, Term::Agg(..)))
                .count();
            let mut g: std::collections::BTreeMap<Vec<Value>, Vec<Vec<Value>>> = Default::default();
            for row in &rel.rows {
                let key = row.key[..group_len].to_vec();
                g.entry(key).or_default().push(row.key[group_len..].to_vec());
            }
            g
        };
        let fns: Vec<crate::intern::AggFn> = c
            .head
            .args
            .iter()
            .filter_map(|t| match t {
                Term::Agg(f, _) => Some(*f),
                _ => None,
            })
            .collect();
        let mut changed = 0usize;
        for (g, rows) in groups {
            let mut out = g.clone();
            for (ai, f) in fns.iter().enumerate() {
                let vals: Vec<i64> = rows
                    .iter()
                    .filter_map(|r| r.get(ai).and_then(|v| v.as_int()))
                    .collect();
                let folded = match f {
                    crate::intern::AggFn::Count => rows.len() as i64,
                    crate::intern::AggFn::Min => vals.iter().min().copied().unwrap_or(0),
                    crate::intern::AggFn::Max => vals.iter().max().copied().unwrap_or(0),
                    crate::intern::AggFn::Sum => vals.iter().sum(),
                };
                out.push(Value::Int(folded));
            }
            // replace any existing row with the same group prefix
            let mut pattern: Vec<Option<Value>> = g.iter().map(|v| Some(*v)).collect();
            for _ in 0..fns.len() {
                pattern.push(None);
            }
            let old: Vec<Vec<Value>> = self
                .relations
                .get(&c.head.pred)
                .map(|r| {
                    r.lookup(&pattern)
                        .into_iter()
                        .map(|id| r.rows[id].key.clone())
                        .collect()
                })
                .unwrap_or_default();
            for o in old {
                if o != out {
                    self.retract(&c.head.pred, &o);
                    changed += 1;
                }
            }
            let exists = self
                .relations
                .get(&c.head.pred)
                .map(|r| r.contains(&out))
                .unwrap_or(false);
            if !exists {
                // witness: the first contributing temp row
                let support = Support::Rule {
                    rule: c
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("agg:{}", c.head.pred)),
                    body: rows
                        .first()
                        .map(|r| {
                            let mut k = g.clone();
                            k.extend(r.iter().copied());
                            (temp.clone(), k)
                        })
                        .into_iter()
                        .collect(),
                };
                let is_new = self
                    .relations
                    .entry(c.head.pred.clone())
                    .or_default()
                    .insert(
                        out.clone(),
                        StoredFact {
                            ann: Ann::unit(),
                            supports: vec![support],
                        },
                    );
                if is_new {
                    self.feed.push(Change::Added(self.epoch, (c.head.pred.clone(), out.clone())));
                    self.change_log.push((self.epoch, (c.head.pred.clone(), out)));
                    changed += 1;
                }
            }
        }
        changed
    }

    /// Read-only iteration over (predicate, relation).
    pub fn relations_iter(&self) -> impl Iterator<Item = (&String, &Relation)> {
        self.relations.iter()
    }

    /// All keys of one relation (read-only convenience).
    pub fn relation_keys(&self, pred: &str) -> Vec<Vec<Value>> {
        self.relations
            .get(pred)
            .map(|r| r.rows.iter().map(|x| x.key.clone()).collect())
            .unwrap_or_default()
    }

    /// Schema summary for rule authoring: each predicate with arity, row
    /// count, and up to three sample facts.
    pub fn schema_summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let mut preds: Vec<&String> = self.relations.keys().collect();
        preds.sort();
        for p in preds {
            let rel = &self.relations[p];
            let arity = rel.rows.first().map(|r| r.key.len()).unwrap_or(0);
            let _ = writeln!(out, "{}/{}: {} rows", p, arity, rel.len());
            for row in rel.rows.iter().take(3) {
                let _ = writeln!(out, "  {}", self.render_fact(p, &row.key));
            }
        }
        out
    }

    /// Installed rule batches: (id, source).
    pub fn batches(&self) -> Vec<(String, String)> {
        self.rule_batches
            .iter()
            .map(|(id, src, _)| (id.clone(), src.clone()))
            .collect()
    }

    /// Uninstall a rule batch (revertable rules). Derived facts are
    /// recomputed on the next `run()` without the removed rules.
    pub fn uninstall(&mut self, id: &str) -> bool {
        let Some(pos) = self.rule_batches.iter().position(|(b, _, _)| b == id) else {
            return false;
        };
        let (_, _, end) = self.rule_batches.remove(pos);
        let start = match pos.checked_sub(1).and_then(|i| self.rule_batches.get(i)) {
            Some((_, _, prev_end)) => *prev_end,
            None => 0,
        };
        if start <= end && end <= self.clauses.len() {
            self.clauses.drain(start..end);
            // renumber the ends of later batches
            let removed = end - start;
            for (_, _, e) in self.rule_batches.iter_mut() {
                if *e > end {
                    *e -= removed;
                }
            }
        }
        self.program_dirty = true;
        true
    }

    /// Program-change recompute: clear every derived relation and reseed
    /// from all base facts, so newly installed rules backfill against the
    /// existing store and uninstalled rules' derivations disappear.
    fn program_recompute(&mut self, strata: &[Vec<usize>]) {
        let mut to_clear: BTreeSet<String> = self.ever_derived.clone();
        for s in strata {
            for &i in s {
                to_clear.insert(self.clauses[i].head.pred.clone());
            }
        }
        for p in &to_clear {
            if let Some(rel) = self.relations.get_mut(p) {
                if rel.len() > 0 {
                    self.feed.push(Change::Cleared(self.epoch, p.clone()));
                }
                rel.clear();
            }
        }
        for rel in self.relations.values_mut() {
            let keys: Vec<Vec<Value>> = rel.rows.iter().map(|r| r.key.clone()).collect();
            rel.pending.extend(keys);
        }
        self.program_dirty = false;
    }

    /// Validate the installed program (parse + stratification).
    pub fn check_program(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.strata()?;
        Ok(())
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Seminaive evaluation of one stratum.
    ///
    /// delta_0 = every predicate's epoch-pending facts (new EDB assertions
    /// plus lower-stratum derivations of this epoch). Each iteration fires
    /// each rule once per positive relational atom, binding that atom's scan
    /// to the current delta and scanning full relations for the rest. Any
    /// new derivation contains at least one delta fact, so this is complete
    /// for set semantics; the fixpoint terminates when an iteration derives
    /// no new facts.
    fn eval_stratum(&mut self, clause_idx: &[usize]) -> usize {
        // aggregation clauses are lowered: the ordinary loop populates
        // their temp relation; the fold step afterwards materializes the
        // aggregated head facts
        let clauses: Vec<Clause> = clause_idx
            .iter()
            .map(|&i| {
                if Self::is_agg_clause(&self.clauses[i]) {
                    self.lower_agg_clause(i)
                } else {
                    self.clauses[i].clone()
                }
            })
            .collect();
        let agg_clauses: Vec<usize> = clause_idx
            .iter()
            .copied()
            .filter(|&i| Self::is_agg_clause(&self.clauses[i]))
            .collect();
        let mut delta: HashMap<String, BTreeSet<Vec<Value>>> = HashMap::new();
        for (pred, rel) in &self.relations {
            if !rel.pending.is_empty() {
                delta.insert(pred.clone(), rel.pending.clone());
            }
        }
        let mut total_new = 0usize;
        loop {
            let mut new_facts: Vec<Key> = Vec::new();
            for clause in &clauses {
                let pos: Vec<usize> = clause
                    .body
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| matches!(l, Lit::Pos(_)))
                    .map(|(i, _)| i)
                    .collect();
                if pos.is_empty() {
                    // no relational atoms: fire at most once per binding of
                    // builtins (which are ground-valued, so once)
                    if let Some(env) = self.solve_builtin_only(clause) {
                        if let Some(k) = self.emit_head(clause, &env) {
                            new_facts.push(k);
                        }
                    }
                    continue;
                }
                for &ai in &pos {
                    self.fire_delta_version(clause, ai, &delta, &mut new_facts);
                }
            }
            if new_facts.is_empty() {
                break;
            }
            total_new += new_facts.len();
            delta.clear();
            for (p, a) in new_facts {
                delta.entry(p).or_default().insert(a);
            }
        }
        for ci in agg_clauses {
            self.fold_agg_clause(ci);
        }

        total_new
    }

    /// Fire one delta version of a rule: body atom `ai` binds against delta
    /// tuples; all other relational atoms scan full relations.
    fn fire_delta_version(
        &mut self,
        clause: &Clause,
        ai: usize,
        delta: &HashMap<String, BTreeSet<Vec<Value>>>,
        out: &mut Vec<Key>,
    ) {
        let atom = match &clause.body[ai] {
            Lit::Pos(a) => a,
            _ => return,
        };
        let delta_keys: Vec<Vec<Value>> = match delta.get(&atom.pred) {
            Some(s) => s.iter().cloned().collect(),
            None => return,
        };
        for k in delta_keys {
            let ann = match self.relations.get(&atom.pred).and_then(|r| r.get(&k)) {
                Some(f) => f.ann.clone(),
                None => continue,
            };
            let mut env = Env::new();
            env.ann = Ann::unit().join(&ann);
            env.body_keys = vec![(atom.pred.clone(), k.clone())];
            if !self.bind_args(&atom.args, &k, &mut env) {
                continue;
            }
            self.solve_rest(clause, 0, Some(ai), &mut env, out);
        }
    }

    fn solve_builtin_only(&mut self, clause: &Clause) -> Option<Env> {
        let mut env = Env::new();
        let mut out = Vec::new();
        // reuse solve_rest with skip = None; it emits heads too, so instead
        // evaluate lits manually via a trick: solve_rest with a sentinel that
        // never matches a body index.
        let impossible_skip = usize::MAX;
        if self.solve_all(clause, 0, impossible_skip, &mut env, &mut out) {
            let _ = out;
            Some(env)
        } else {
            None
        }
    }

    /// Resolve body lits from index `i` onward. `skip` bypasses an atom
    /// (already bound by the delta version). Emits heads; returns whether at
    /// least one solution was found.
    fn solve_rest(
        &mut self,
        clause: &Clause,
        i: usize,
        skip: Option<usize>,
        env: &mut Env,
        out: &mut Vec<Key>,
    ) -> bool {
        self.solve_all(clause, i, skip.unwrap_or(usize::MAX), env, out)
    }

    fn solve_all(
        &mut self,
        clause: &Clause,
        i: usize,
        skip: usize,
        env: &mut Env,
        out: &mut Vec<Key>,
    ) -> bool {
        if i == clause.body.len() {
            if let Some(k) = self.emit_head(clause, env) {
                out.push(k);
                return true;
            }
            return false;
        }
        if i == skip {
            return self.solve_all(clause, i + 1, skip, env, out);
        }
        match &clause.body[i] {
            Lit::Pos(atom) => {
                let bound = self.resolve_bound(&atom.args, env);
                let ids = match self.relations.get(&atom.pred) {
                    Some(rel) => rel.lookup(&bound),
                    None => Vec::new(),
                };
                let mut any = false;
                let outer = env.mark();
                for id in ids {
                    // fetch under a fresh immutable borrow; ids stay valid
                    // because rows are never removed mid-stratum (retraction
                    // only happens between runs)
                    let (k, ann) = {
                        let rel = match self.relations.get(&atom.pred) {
                            Some(r) => r,
                            None => continue,
                        };
                        let row = &rel.rows[id];
                        (row.key.clone(), row.fact.ann.clone())
                    };
                    let mark = env.mark();
                    env.ann = env.ann.join(&ann);
                    env.body_keys.push((atom.pred.clone(), k.clone()));
                    if self.bind_args(&atom.args, &k, env)
                        && self.solve_all(clause, i + 1, skip, env, out)
                    {
                        any = true;
                    }
                    env.undo(&mark);
                }
                env.undo(&outer);
                any
            }
            Lit::Neg(atom) => {
                // negation-as-absence against the full relation; the stratum
                // is already complete for lower-stratum predicates
                if self.matches_any(atom, env) {
                    return false;
                }
                self.solve_all(clause, i + 1, skip, env, out)
            }
            Lit::Now(t) => match t {
                Term::Wildcard => self.solve_all(clause, i + 1, skip, env, out),
                Term::Int(v) => {
                    if *v == self.now {
                        self.solve_all(clause, i + 1, skip, env, out)
                    } else {
                        false
                    }
                }
                Term::Agg(..) => false, // aggregates never appear in bodies
                Term::Var(v) => match env.map.get(v) {
                    Some(Value::Int(x)) if *x == self.now => {
                        self.solve_all(clause, i + 1, skip, env, out)
                    }
                    Some(_) => false,
                    None => {
                        env.bind(v, Value::Int(self.now));
                        self.solve_all(clause, i + 1, skip, env, out)
                    }
                },
                Term::Sym(_) => false,
            },
            Lit::Cmp(op, a, b) => {
                use crate::ast::Expr;
                let av = self.resolve_term(a, env);
                // plain-term RHS (covers symbol equality) first
                if let Expr::T(bt) = b {
                    match (av, self.resolve_term(bt, env)) {
                        (Some(av), Some(bv)) => {
                            return if cmp_holds(*op, av, bv) {
                                self.solve_all(clause, i + 1, skip, env, out)
                            } else {
                                false
                            };
                        }
                        (None, Some(bv)) => {
                            if let Term::Var(v) = a {
                                env.bind(v, bv);
                                return self.solve_all(clause, i + 1, skip, env, out);
                            }
                            return false;
                        }
                        _ => {}
                    }
                }
                // arithmetic path: expr = coeff * X + const
                if let Some((coeff, c, var)) = linearize(b, env) {
                    match (av, coeff) {
                        (Some(av), 0) => {
                            return if cmp_holds(*op, av, Value::Int(c)) {
                                self.solve_all(clause, i + 1, skip, env, out)
                            } else {
                                false
                            };
                        }
                        (Some(av), 1) => {
                            // RHS is X + c; solvable for equality: av = X + c
                            if *op == CmpOp::Eq && var.is_some() {
                                if let Some(ai) = av.as_int() {
                                    let v = var.unwrap();
                                    env.bind(&v, Value::Int(ai - c));
                                    return self.solve_all(clause, i + 1, skip, env, out);
                                }
                            }
                        }
                        (None, 0) => {
                            if let Term::Var(v) = a {
                                env.bind(v, Value::Int(c));
                                return self.solve_all(clause, i + 1, skip, env, out);
                            }
                        }
                        _ => {}
                    }
                }
                false
            }
        }
    }

    /// Resolve a pattern's terms against env into concrete bound values
    /// (None = unbound). Unknown symbols resolve to a sentinel that matches
    /// nothing, so lookups short-circuit.
    fn resolve_bound(&self, pat: &[Term], env: &Env) -> Vec<Option<Value>> {
        pat.iter()
            .map(|t| match t {
                Term::Int(i) => Some(Value::Int(*i)),
                Term::Agg(..) => None, // never matches: aggregates are heads-only
                Term::Var(v) => env.map.get(v).copied(),
                Term::Sym(s) => match self.interner.lookup(s) {
                    Some(sym) => Some(Value::Sym(sym)),
                    None => Some(Value::Int(i64::MIN)), // unknown: matches nothing
                },
                Term::Wildcard => None,
            })
            .collect()
    }

    fn matches_any(&self, atom: &crate::ast::Atom, env: &Env) -> bool {
        let bound = self.resolve_bound(&atom.args, env);
        let ids = match self.relations.get(&atom.pred) {
            Some(rel) => rel.lookup(&bound),
            None => return false,
        };
        for id in ids {
            let k = self.relations[&atom.pred].rows[id].key.clone();
            let mut e = env.clone();
            if self.bind_args(&atom.args, &k, &mut e) {
                return true;
            }
        }
        false
    }

    fn resolve_term(&self, t: &Term, env: &Env) -> Option<Value> {
        match t {
            Term::Var(_) => env.lookup(t),
            Term::Int(i) => Some(Value::Int(*i)),
            Term::Sym(s) => self.interner.lookup(s).map(Value::Sym),
            Term::Wildcard => None,
            Term::Agg(..) => None, // heads-only; never resolved in bodies
        }
    }

    fn bind_args(&self, pat: &[Term], args: &[Value], env: &mut Env) -> bool {
        if pat.len() != args.len() {
            return false;
        }
        for (p, a) in pat.iter().zip(args) {
            let ok = match p {
                Term::Wildcard => true,
                Term::Agg(..) => false,
                Term::Var(v) => match env.map.get(v) {
                    Some(bound) => *bound == *a,
                    None => {
                        env.bind(v, *a);
                        true
                    }
                },
                Term::Int(i) => Value::Int(*i) == *a,
                Term::Sym(s) => matches!((self.interner.lookup(s), a),
                    (Some(sv), Value::Sym(av)) if sv == *av),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    fn ground_args(&mut self, pat: &[Term]) -> Option<Vec<Value>> {
        pat.iter()
            .map(|p| match p {
                Term::Sym(s) => Some(Value::Sym(self.interner.intern(s))),
                Term::Int(i) => Some(Value::Int(*i)),
                _ => None,
            })
            .collect()
    }

    fn emit_head(&mut self, clause: &Clause, env: &Env) -> Option<Key> {
        let mut args = Vec::with_capacity(clause.head.args.len());
        for t in &clause.head.args {
            let v = match t {
                Term::Sym(s) => Value::Sym(self.interner.intern(s)),
                Term::Int(i) => Value::Int(*i),
                Term::Var(v) => *env.map.get(v)?,
                Term::Wildcard => return None, // safety: no unbound heads
                Term::Agg(..) => return None,  // agg rules are lowered, never emitted
            };
            args.push(v);
        }
        let support = Support::Rule {
            rule: clause
                .name
                .clone()
                .unwrap_or_else(|| format!("rule/{}", clause.head.pred)),
            body: env.body_keys.clone(),
        };
        // fast path: duplicate emission — merge + capped witness recording
        if let Some(rel) = self.relations.get_mut(&clause.head.pred) {
            if rel.merge_ann(&args, &env.ann, support.clone()) {
                return None;
            }
        }
        let rel = self.relations.entry(clause.head.pred.clone()).or_default();
        let key = (clause.head.pred.clone(), args.clone());
        let is_new = rel.insert(
            key.1.clone(),
            StoredFact {
                ann: env.ann.clone(),
                supports: vec![support],
            },
        );
        if is_new {
            let out = key.clone();
            self.feed.push(Change::Added(self.epoch, key));
            self.change_log.push((self.epoch, out.clone()));
            Some(out)
        } else {
            None
        }
    }

    // -------------------------------------------------------------- queries

    /// Query a predicate with a pattern; `None` slots are wildcards.
    /// Index-aware: bound positions select the smallest bucket instead of
    /// scanning the relation.
    pub fn query(&self, pred: &str, pattern: &[Option<Value>]) -> Vec<(Vec<Value>, Ann)> {
        let mut out = Vec::new();
        let Some(rel) = self.relations.get(pred) else {
            return out;
        };
        let ids: Vec<usize> = if pattern.iter().any(|p| p.is_some()) {
            rel.lookup(pattern)
        } else {
            (0..rel.rows.len()).collect()
        };
        for id in ids {
            let row = &rel.rows[id];
            if !pattern.is_empty() && pattern.len() != row.key.len() {
                return Vec::new();
            }
            if pattern
                .iter()
                .zip(&row.key)
                .all(|(p, v)| p.as_ref().map(|pv| pv == v).unwrap_or(true))
            {
                out.push((row.key.clone(), row.fact.ann.clone()));
            }
        }
        out
    }

    pub fn fact(&self, pred: &str, args: &[Value]) -> Option<StoredFact> {
        self.relations.get(pred).and_then(|r| r.get(args)).cloned()
    }

    /// Demand-driven query (magic sets): answers a goal like
    /// `reports_to("n0", Y)` WITHOUT materializing the full fixpoint of
    /// `reports_to`. The rewritten demand program is installed, evaluated,
    /// and removed; the base store is untouched. `last_demand_facts`
    /// records how many facts the demand evaluation derived (the
    /// demand-relevant slice) for observability.
    pub fn ask_deep(&mut self, goal: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let prog = crate::ast::parse_program(&format!("{goal}."))?;
        if prog.len() != 1 || !prog[0].is_fact {
            return Err(crate::ast::ParseError(format!(
                "ask_deep: expected a single atom, got {goal:?}"
            ))
            .into());
        }
        let head = &prog[0].head;
        // goal over an EDB predicate: no rewriting needed
        if !self.clauses.iter().any(|c| c.head.pred == head.pred) {
            return Ok(self.ask(goal)?);
        }
        // goal over a materialized derived predicate: read the store
        // directly (the demand machinery is for unmaterialized predicates;
        // callers keep materialization fresh via run/maintain)
        if self
            .relations
            .get(&head.pred)
            .map(|r| r.len() > 0)
            .unwrap_or(false)
        {
            return Ok(self.ask(goal)?);
        }
        let materialized: std::collections::BTreeSet<String> = self
            .relations
            .iter()
            .filter(|(_, rel)| rel.len() > 0)
            .map(|(p, _)| p.clone())
            .collect();
        let demand = crate::magic::build(&self.clauses, head, &materialized)?;
        let relations_before: std::collections::BTreeSet<String> =
            self.relations.keys().cloned().collect();
        let saved_clauses = std::mem::replace(&mut self.clauses, demand.clauses);
        self.check_program()?;
        self.run();
        // stats: facts created by the demand program
        let mut created = 0usize;
        for name in self.relations.keys() {
            if !relations_before.contains(name) {
                created += self.relations[name].len();
            }
        }
        self.last_demand_facts = created;
        // answer from the adorned predicate
        let answer_pred = demand.answer_pred;
        let rows = {
            let vars: Vec<&String> = head
                .args
                .iter()
                .filter_map(|t| match t {
                    Term::Var(v) => Some(v),
                    _ => None,
                })
                .collect();
            let mut rows = Vec::new();
            if let Some(rel) = self.relations.get(&answer_pred) {
                for row in &rel.rows {
                    let mut env = Env::new();
                    if self.bind_args(&head.args, &row.key, &mut env) {
                        let row_str: Vec<String> = vars
                            .iter()
                            .map(|v| {
                                let val = env
                                    .map
                                    .get(*v)
                                    .map(|x| self.interner.display(x))
                                    .unwrap_or_else(|| "_".into());
                                format!("{}={}", v, val)
                            })
                            .collect();
                        rows.push(row_str.join(", "));
                    }
                }
            }
            if vars.is_empty() && !rows.is_empty() {
                rows = vec![String::new()];
            }
            rows
        };
        // cleanup: restore the base program, drop demand-only relations
        self.clauses = saved_clauses;
        self.relations
            .retain(|name, _| relations_before.contains(name));
        Ok(rows)
    }

    /// Ask a conjunctive-free ground-or-varied atom, e.g.
    /// `current("alice", R, O)` or `reports_to(X, "carol")`, and get the
    /// variable bindings. Read-only, side-effect-free, terminating: the
    /// agent-facing query surface. Returns rendered rows like
    /// `R=works_at, O=acme`; a ground goal yields one empty-string row if
    /// the fact holds.
    pub fn ask(&self, goal: &str) -> Result<Vec<String>, crate::ast::ParseError> {
        let prog = crate::ast::parse_program(&format!("{goal}."))?;
        if prog.len() != 1 || !prog[0].is_fact {
            return Err(crate::ast::ParseError(format!(
                "ask: expected a single atom, got {goal:?}"
            )));
        }
        let head = &prog[0].head;
        let vars: Vec<&String> = head
            .args
            .iter()
            .filter_map(|t| match t {
                Term::Var(v) => Some(v),
                _ => None,
            })
            .collect();
        let mut rows = Vec::new();
        if let Some(rel) = self.relations.get(&head.pred) {
            let bound = self.resolve_bound(&head.args, &Env::new());
            let ids = if bound.iter().any(|b| b.is_some()) {
                rel.lookup(&bound)
            } else {
                (0..rel.rows.len()).collect()
            };
            for id in ids {
                let k = &rel.rows[id].key;
                let mut env = Env::new();
                if self.bind_args(&head.args, k, &mut env) {
                    let row: Vec<String> = vars
                        .iter()
                        .map(|v| {
                            let val = env
                                .map
                                .get(*v)
                                .map(|x| self.interner.display(x))
                                .unwrap_or_else(|| "_".into());
                            format!("{}={}", v, val)
                        })
                        .collect();
                    rows.push(row.join(", "));
                }
            }
        }
        if vars.is_empty() && !rows.is_empty() {
            rows = vec![String::new()];
        }
        Ok(rows)
    }

    // -------------------------------------------------------------- why

    /// Render a proof tree for a fact (cycle-safe).
    pub fn why(&self, pred: &str, args: &[Value]) -> String {
        let mut out = String::new();
        self.why_rec(pred, args, 0, &mut out, &mut BTreeSet::new());
        out
    }

    fn why_rec(
        &self,
        pred: &str,
        args: &[Value],
        indent: usize,
        out: &mut String,
        seen: &mut BTreeSet<Key>,
    ) {
        let pad = "  ".repeat(indent);
        let rendered = self.render_fact(pred, args);
        let f = match self.relations.get(pred).and_then(|r| r.get(args)) {
            Some(f) => f.clone(),
            None => {
                let _ = writeln!(out, "{pad}{rendered}  [unknown fact]");
                return;
            }
        };
        if !seen.insert((pred.to_string(), args.to_vec())) {
            let _ = writeln!(out, "{pad}{rendered}  [cycle: shown above]");
            return;
        }
        let prov: Vec<&String> = f.ann.prov.iter().collect();
        let _ = writeln!(
            out,
            "{pad}{rendered}  (conf {:.3}, prov {prov:?})",
            f.ann.conf
        );
        // render one witness derivation per distinct rule (further
        // witnesses of the same rule add no explanatory power)
        let mut shown_rules: BTreeSet<&str> = BTreeSet::new();
        let mut shown_any = false;
        for s in &f.supports {
            match s {
                Support::Base => {
                    if shown_rules.insert("«base»") {
                        let _ = writeln!(out, "{pad}  \u{21b3} asserted (base fact)");
                        shown_any = true;
                    }
                }
                Support::Rule { rule, body } => {
                    if shown_rules.insert(rule.as_str()) {
                        let _ = writeln!(out, "{pad}  \u{21b3} via {rule}");
                        for (p, a) in body {
                            self.why_rec(p, a, indent + 2, out, seen);
                        }
                        shown_any = true;
                    }
                }
            }
        }
        if !shown_any && !f.supports.is_empty() {
            let _ = writeln!(out, "{pad}  \u{21b3} (witnesses already shown above)");
        }
    }

    pub fn render_fact(&self, pred: &str, args: &[Value]) -> String {
        let rendered: Vec<String> = args.iter().map(|v| self.interner.display(v)).collect();
        format!("{}({})", pred, rendered.join(", "))
    }
}

/// Linearize an additive integer expression around its unique unbound
/// variable: returns (coeff, const, Some(var)) where expr = coeff*var+const,
/// or (0, value, None) when fully bound. None when not linear / two unbound
/// variables / symbol-typed.
fn linearize(e: &crate::ast::Expr, env: &Env) -> Option<(i64, i64, Option<String>)> {
    use crate::ast::Expr;
    match e {
        Expr::T(Term::Int(i)) => Some((0, *i, None)),
        Expr::T(Term::Var(v)) => match env.map.get(v) {
            Some(Value::Int(x)) => Some((0, *x, None)),
            None => Some((1, 0, Some(v.clone()))),
            Some(Value::Sym(_)) => None,
        },
        Expr::T(_) => None,
        Expr::Add(a, b) => {
            let (c1, k1, v1) = linearize(a, env)?;
            let (c2, k2, v2) = linearize(b, env)?;
            combine(v1, v2, c1 + c2, k1 + k2)
        }
        Expr::Sub(a, b) => {
            let (c1, k1, v1) = linearize(a, env)?;
            let (c2, k2, v2) = linearize(b, env)?;
            combine(v1, v2, c1 - c2, k1 - k2)
        }
    }
}

fn combine(
    v1: Option<String>,
    v2: Option<String>,
    c: i64,
    k: i64,
) -> Option<(i64, i64, Option<String>)> {
    let v = match (v1, v2) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(_), Some(_)) => return None, // two unbound vars: not linear
    };
    Some((c, k, v))
}

/// DFS reachability over the IDB dependency graph.
fn reaches(deps: &HashMap<&str, Vec<(&str, bool)>>, from: &str, to: &str) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        if n == to {
            return true;
        }
        if !seen.insert(n) {
            continue;
        }
        if let Some(edges) = deps.get(n) {
            for (next, _) in edges {
                stack.push(next);
            }
        }
    }
    false
}

fn cmp_holds(op: CmpOp, a: Value, b: Value) -> bool {    match op {
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
    }
}
