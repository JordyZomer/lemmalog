//! Hybrid retrieval for context assembly: BM25 keyword scoring over facts
//! and episodes, entity-match boosting (the graph half — a query that names
//! an entity pulls that entity's facts), and budget-aware selection that
//! fills distilled facts first and their provenance episodes last.
//!
//! This replaces "dump everything" context assembly: the measured failure
//! mode was context bloat (the token advantage over transcripts compressed
//! from 4-12x to 1-5x as extraction got richer) — selection, not
//! extraction, is the bottleneck at the context boundary.

use crate::agent::Episode;
use crate::eval::Engine;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Public re-export for callers needing the same tokenization (e.g. the
/// benchmark runner's subject-question matcher).
pub fn tokenize_pub(s: &str) -> Vec<String> {
    tokenize(s)
}

fn tokenize(s: &str) -> Vec<String> {
    // Common function words: no topical signal, but they match everywhere
    // once relation names split on `_` (`works_at` -> `works`, `at`),
    // poisoning BM25 with false overlaps
    const STOP: [&str; 22] = [
        "a", "an", "the", "at", "in", "on", "of", "to", "is", "are", "was",
        "were", "be", "do", "does", "did", "who", "what", "which", "for",
        "with", "and",
    ];
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1 && !STOP.contains(t))
        .map(|t| t.to_string())
        .collect()
}

/// Okapi BM25 over a fixed document set (no dependencies; agent-memory
/// scale makes rebuild-per-query fine).
pub struct Bm25 {
    docs: Vec<Vec<String>>,
    df: HashMap<String, usize>,
    avg_len: f64,
}

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

impl Bm25 {
    pub fn new(texts: &[String]) -> Self {
        let docs: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t)).collect();
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in &docs {
            for t in doc.iter().collect::<BTreeSet<_>>() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
        }
        let avg_len = if docs.is_empty() {
            1.0
        } else {
            docs.iter().map(|d| d.len() as f64).sum::<f64>() / docs.len() as f64
        };
        Bm25 { docs, df, avg_len }
    }

    fn idf(&self, term: &str) -> f64 {
        let n = self.docs.len() as f64;
        let df = *self.df.get(term).unwrap_or(&0) as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Scores for every document (callers rank/fill).
    pub fn scores(&self, query: &str) -> Vec<f64> {
        let q: BTreeSet<String> = tokenize(query).into_iter().collect();
        self.docs
            .iter()
            .map(|doc| {
                if doc.is_empty() || q.is_empty() {
                    return 0.0;
                }
                let len = doc.len() as f64;
                let tf: HashMap<&String, usize> = {
                    let mut m = HashMap::new();
                    for t in doc {
                        *m.entry(t).or_insert(0) += 1;
                    }
                    m
                };
                q.iter()
                    .map(|term| {
                        let f = *tf.get(term).unwrap_or(&0) as f64;
                        if f == 0.0 {
                            return 0.0;
                        }
                        self.idf(term) * f * (BM25_K1 + 1.0)
                            / (f + BM25_K1 * (1.0 - BM25_B + BM25_B * len / self.avg_len))
                    })
                    .sum()
            })
            .collect()
    }
}

/// Internal bookkeeping relations that should not appear in retrieved
/// context (canonicalization plumbing, aggregation temps, seeding).
fn internal_pred(p: &str) -> bool {
    p == "entity"
        || p == "alias"
        || p == "aliased"
        || p == "same_as"
        || p == "maps_to"
        || p == "alias_conflict"
        || p == "describes"
        || p == "mentions"
        || p == "edge"
        || p.starts_with("__agg:")
        || p.starts_with("__magic")
        || p.starts_with("_magic")
}

struct FactDoc {
    render: String,
    /// entity names appearing in this fact (for match boosting)
    entities: Vec<String>,
    prov: BTreeSet<String>,
}

pub struct Retrieval {
    facts: Vec<FactDoc>,
    fact_bm25: Bm25,
    ep_bm25: Bm25,
    episodes: Vec<Episode>,
    /// entity name -> fact indices mentioning it (the graph adjacency)
    by_entity: BTreeMap<String, Vec<usize>>,
    /// 1-hop neighbours: entity -> entities sharing a fact
    neighbors: BTreeMap<String, BTreeSet<String>>,
}

/// What the selector chose: rendered fact lines (scored, budgeted) and the
/// episode indices to include verbatim.
pub struct Selection {
    pub fact_lines: Vec<(String, f64)>,
    pub episodes: Vec<usize>,
    pub budget_tokens: usize,
    pub used_tokens: usize,
}

impl Retrieval {
    /// Build the index from the engine's current state and the episode
    /// log. O(facts + episodes); rebuild per query is fine at agent scale.
    pub fn build(engine: &Engine, episodes: &[Episode]) -> Self {
        let mut facts: Vec<FactDoc> = Vec::new();
        let mut by_entity: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut co_occurrence: BTreeMap<(String, String), ()> = BTreeMap::new();
        let mut preds: Vec<&String> = engine.relations.keys().collect();
        preds.sort();
        for p in &preds {
            if internal_pred(p) {
                continue;
            }
            for key in engine.relation_keys(p) {
                let mut entities = Vec::new();
                // position 1 of arity>=3 facts holds the relation NAME
                // (current(S,R,O), edge(S,R,O,VF,VT,TS)): matchable
                // entities but must not BRIDGE subjects — otherwise
                // "works_at" neighbors every works_at fact's subject and
                // the hop boost leaks irrelevant subjects
                let bridge: Vec<String> = key
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !(key.len() >= 3 && *i == 1))
                    .filter_map(|(_, v)| match v {
                        crate::intern::Value::Sym(s) => {
                            Some(engine.interner.resolve(*s).to_string())
                        }
                        _ => None,
                    })
                    .collect();
                for v in &key {
                    if let crate::intern::Value::Sym(s) = v {
                        entities.push(engine.interner.resolve(*s).to_string());
                    }
                }
                let prov = engine
                    .fact(p, &key)
                    .map(|f| f.ann.prov)
                    .unwrap_or_default();
                let idx = facts.len();
                for e in &entities {
                    by_entity.entry(e.clone()).or_default().push(idx);
                }
                for a in &bridge {
                    for b in &bridge {
                        if a != b {
                            co_occurrence.insert((a.clone(), b.clone()), ());
                        }
                    }
                }
                facts.push(FactDoc {
                    render: engine.render_fact(p, &key),
                    entities,
                    prov,
                });
            }
        }
        let mut neighbors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (a, b) in co_occurrence.keys() {
            neighbors
                .entry(a.clone())
                .or_default()
                .insert(b.clone());
        }
        let fact_texts: Vec<String> = facts.iter().map(|f| f.render.clone()).collect();
        let ep_texts: Vec<String> = episodes.iter().map(|e| e.text.clone()).collect();
        Retrieval {
            fact_bm25: Bm25::new(&fact_texts),
            ep_bm25: Bm25::new(&ep_texts),
            facts,
            episodes: episodes.to_vec(),
            by_entity,
            neighbors,
        }
    }

    /// Entity names from the query: exact word match against known names
    /// (case-insensitive). This is the graph entry point.
    fn query_entities(&self, query: &str) -> (BTreeSet<String>, BTreeSet<String>) {
        let q = query.to_lowercase();
        let mut direct = BTreeSet::new();
        for name in self.by_entity.keys() {
            let ln = name.to_lowercase();
            if ln.len() >= 2 && q.contains(&ln) {
                direct.insert(name.clone());
            }
        }
        let mut hops = BTreeSet::new();
        for d in &direct {
            if let Some(ns) = self.neighbors.get(d) {
                for n in ns {
                    if !direct.contains(n) {
                        hops.insert(n.clone());
                    }
                }
            }
        }
        (direct, hops)
    }

    /// Score, rank, and budget: facts by BM25 + entity boosts; then the
    /// selected facts' provenance episodes plus the top BM25 episodes fill
    /// the remaining budget.
    pub fn select(&self, query: &str, budget_tokens: usize) -> Selection {
        let bm25 = self.fact_bm25.scores(query);
        let (direct, hops) = self.query_entities(query);
        let mut max_bm25 = 0.0f64;
        for s in &bm25 {
            max_bm25 = max_bm25.max(*s);
        }
        let norm = if max_bm25 > 0.0 { max_bm25 } else { 1.0 };

        let mut scored: Vec<(usize, f64)> = self
            .facts
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut s = bm25[i] / norm;
                for e in &f.entities {
                    if direct.contains(e) {
                        s += 1.5; // the query names this entity
                    } else if hops.contains(e) {
                        s += 0.4; // one hop from a named entity
                    }
                }
                (i, s)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let fact_budget = budget_tokens.saturating_mul(4) * 6 / 10;
        let mut fact_lines = Vec::new();
        let mut used = 0usize;
        let mut wanted_eps: BTreeSet<String> = BTreeSet::new();
        for (i, s) in &scored {
            let line = format!("{}\n", self.facts[*i].render);
            if used + line.len() > fact_budget {
                break;
            }
            used += line.len();
            wanted_eps.extend(self.facts[*i].prov.iter().cloned());
            fact_lines.push((self.facts[*i].render.clone(), *s));
        }

        // episodes: provenance of selected facts, then BM25 relevance
        let ep_scores = self.ep_bm25.scores(query);
        let mut ep_ranked: Vec<(usize, f64, bool)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| {
                let prov = wanted_eps.contains(&ep.id);
                (i, ep_scores.get(i).copied().unwrap_or(0.0), prov)
            })
            .collect();
        ep_ranked.sort_by(|a, b| {
            (b.2, b.1).partial_cmp(&(a.2, a.1)).unwrap_or(std::cmp::Ordering::Equal)
        });

        let ep_budget = budget_tokens.saturating_mul(4).saturating_sub(used);
        let mut chosen_eps = Vec::new();
        let mut ep_used = 0usize;
        for (i, _, _) in ep_ranked {
            if chosen_eps.len() >= 8 {
                break;
            }
            let block_len = self.episodes[i].text.len() + 24;
            if ep_used + block_len > ep_budget {
                // try to fit a truncation of the episode
                let remaining = ep_budget.saturating_sub(ep_used);
                if remaining > 400 {
                    // mark for truncation by including and letting render cut
                    ep_used += remaining;
                    chosen_eps.push(i);
                }
                continue;
            }
            ep_used += block_len;
            chosen_eps.push(i);
        }

        Selection {
            fact_lines,
            episodes: chosen_eps,
            budget_tokens,
            used_tokens: (used + ep_used) / 4,
        }
    }

    /// Render the two-section context: distilled facts (relevance-ordered)
    /// at the top, verbatim episodes at the bottom — positional assembly
    /// (lost-in-the-middle mitigation) with selection doing the trimming.
    pub fn render(&self, sel: &Selection) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "== memory (relevance-selected) ==");
        for (line, _) in &sel.fact_lines {
            let _ = writeln!(out, "{line}");
        }
        if !sel.episodes.is_empty() {
            let _ = writeln!(out, "\n== source episodes (verbatim) ==");
            let ep_budget = sel.budget_tokens.saturating_mul(4).saturating_sub(sel.fact_lines.iter().map(|(l, _)| l.len() + 1).sum::<usize>());
            let mut used = 0usize;
            for &i in &sel.episodes {
                let ep = &self.episodes[i];
                let mut block = format!("[{}] {}\n", ep.id, ep.text);
                if used + block.len() > ep_budget {
                    let remaining = ep_budget.saturating_sub(used);
                    if remaining > 200 {
                        // char-safe truncation: byte slicing can split UTF-8
                        let cut: String =
                            ep.text.chars().take(remaining.min(ep.text.len())).collect();
                        block = format!("[{}] {}…\n", ep.id, cut);
                    } else {
                        break;
                    }
                }
                used += block.len();
                out.push_str(&block);
            }
        }
        out
    }
}
