//! The LLM integration layer (Phase 4 of the design).
//!
//! The fixpoint never contains an LLM call: extraction happens at the
//! *ingestion boundary* (the [`Extractor`] trait — a real deployment plugs an
//! OpenIE LLM here, tests and examples use [`MockExtractor`]), update
//! decisions are deterministic rules first (Mem0-style
//! ADD/UPDATE/NOOP/escalate), and derivation runs asynchronously via
//! `maintain()`. The [`ContextAssembler`] places distilled facts at the top
//! of the window and verbatim provenance at the bottom (lost-in-the-middle
//! mitigation) under a token budget.

use crate::eval::{Ann, Engine};
use crate::intern::Value;
use crate::intern::Term;
use std::collections::HashMap;
use std::fmt::Write as _;

/// A conversational episode: the unit of ingestion and provenance.
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: String,
    pub text: String,
    pub ts: i64,
    /// The identified speaker, when the application knows it: first-person
    /// references in the episode resolve to this entity during extraction.
    pub speaker: Option<String>,
}

/// One candidate fact produced by extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFact {
    pub subj: String,
    pub pred: String,
    pub obj: String,
    pub confidence: f64,
}

/// The extraction boundary. Implementations call an LLM in production;
/// they must be memoizable by (episode, extractor-version).
pub trait Extractor {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact>;

    /// Observability: (model calls, failures). Defaults to zero for
    /// deterministic extractors.
    fn stats(&self) -> (usize, usize) {
        (0, 0)
    }
}

/// Boxed extractors are extractors: lets callers swap implementations
/// (live vs file-cached) behind one `AgentMemory` type.
impl Extractor for Box<dyn Extractor> {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        (**self).extract(episode)
    }

    fn stats(&self) -> (usize, usize) {
        (**self).stats()
    }
}

/// Deterministic stand-in for the LLM OpenIE step: parses `S --rel--> O`
/// lines at fixed confidence. Used by tests and examples.
pub struct MockExtractor {
    pub confidence: f64,
    seen: HashMap<String, Vec<CandidateFact>>,
}

impl MockExtractor {
    pub fn new(confidence: f64) -> Self {
        MockExtractor {
            confidence,
            seen: HashMap::new(),
        }
    }
}

impl Extractor for MockExtractor {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        // memoized by episode id: never re-extracted
        if let Some(cached) = self.seen.get(&episode.id) {
            return cached.clone();
        }
        let out = parse_protocol(episode.text.as_str(), self.confidence);
        self.seen.insert(episode.id.clone(), out.clone());
        out
    }
}

/// Why an entity token fails strict validation, as a self-correcting
/// reason (echoed to the model that produced it).
fn entity_token_problem(s: &str) -> Option<String> {
    // unresolved-reference words: pronouns and role placeholders that mean
    // the model failed to resolve the entity
    const BLOCKED: [&str; 14] = [
        "i", "me", "my", "mine", "speaker", "user", "they", "them", "he", "she",
        "it", "we", "you", "that",
    ];
    let lower = s.to_lowercase();
    if s.is_empty() {
        Some("empty entity name".to_string())
    } else if BLOCKED.contains(&lower.as_str()) {
        Some(format!(
            "'{s}' is a pronoun or role word — resolve it to the entity's real name"
        ))
    } else if s.len() > 60 || s.split_whitespace().count() > 8 {
        Some("looks like prose (more than 8 words) — entity names are short".to_string())
    } else if !s
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '\'' | ' ' | '/' | '.' | ':' | '#'))
    {
        Some("contains punctuation or prose characters — entity names use letters, digits, '_', '-', apostrophes, and '/', '.', ':', '#' in source references".to_string())
    } else if s.contains(' ') && s.chars().any(|c| matches!(c, '/' | '.' | ':' | '#')) {
        // Path characters are allowed so that source references
        // (`src/agent.rs:118`) can be entities, but only as single tokens:
        // punctuation *plus* spaces is the signature of leaked prose, which
        // is exactly what strict validation exists to drop.
        Some("looks like prose — source references must not contain spaces (`src/agent.rs:118`, not `see src/agent.rs, line 118`)".to_string())
    } else {
        None
    }
}

fn valid_entity_token(s: &str) -> bool {
    entity_token_problem(s).is_none()
}

/// Strict protocol parsing for MODEL output: lines that are not exactly
/// `Entity --relation[conf]--> Entity` with clean entity tokens are
/// dropped. Reasoning models sometimes leak deliberation into the answer;
/// those lines (questions, prose, bullets) must not become facts.
pub fn parse_protocol_strict(text: &str, default_confidence: f64) -> Vec<CandidateFact> {
    parse_protocol(text, default_confidence)
        .into_iter()
        .filter(|c| {
            valid_entity_token(&c.subj)
                && valid_entity_token(&c.obj)
                && !c.pred.is_empty()
                && c.pred
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        })
        .collect()
}

/// One line of the protocol, or a reason it cannot parse.
fn parse_line(raw: &str, default_confidence: f64) -> Result<CandidateFact, String> {
    let line = raw.trim();
    let (s, rest) = line
        .split_once("--")
        .ok_or_else(|| "no `--rel-->` structure".to_string())?;
    let (rel, o) = rest
        .split_once("-->")
        .ok_or_else(|| "has `--` but no `-->`".to_string())?;
    // optional confidence suffix on the relation: `rel[0.8]`
    let (rel, conf) = match rel.trim().rsplit_once('[') {
        Some((r, c)) if c.ends_with(']') => (
            r.trim(),
            c.trim_end_matches(']')
                .trim()
                .parse::<f64>()
                .unwrap_or(default_confidence),
        ),
        _ => (rel.trim(), default_confidence),
    };
    if rel.is_empty() {
        return Err("empty relation".to_string());
    }
    Ok(CandidateFact {
        subj: s.trim().to_string(),
        pred: rel.to_string(),
        obj: o.trim().to_string(),
        confidence: conf,
    })
}

/// Line protocol shared by mock and LLM extractors: each line is
/// `S --rel--> O` with optional per-fact confidence `S --rel[0.8]--> O`.
/// Unparseable lines are skipped (extraction is best-effort).
pub fn parse_protocol(text: &str, default_confidence: f64) -> Vec<CandidateFact> {
    text.lines()
        .filter_map(|l| parse_line(l, default_confidence).ok())
        .collect()
}

/// Strict parse WITH a drop report: `(facts, dropped)` where dropped is
/// `(line, reason)` for every line not asserted — parse failures and
/// strict-validation failures alike. Silent zero-fact ingestion is the
/// worst failure mode a caller can face; this makes it loud.
pub fn parse_protocol_reported(
    text: &str,
    default_confidence: f64,
) -> (Vec<CandidateFact>, Vec<(String, String)>) {
    let mut facts = Vec::new();
    let mut dropped = Vec::new();
    for raw in text.lines() {
        if raw.trim().is_empty() {
            continue;
        }
        match parse_line(raw, default_confidence) {
            Ok(c) => {
                let problem = entity_token_problem(&c.subj)
                    .or_else(|| entity_token_problem(&c.obj))
                    .or_else(|| {
                        if c.pred
                            .chars()
                            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
                        {
                            None
                        } else {
                            Some("relation must be lower snake_case (e.g. works_at)".to_string())
                        }
                    });
                match problem {
                    Some(reason) => dropped.push((raw.trim().to_string(), reason)),
                    None => facts.push(c),
                }
            }
            Err(reason) => dropped.push((raw.trim().to_string(), reason)),
        }
    }
    (facts, dropped)
}

/// An [`Extractor`] whose extraction step is a caller-supplied model call —
/// bring your own provider (OpenAI, Anthropic, a local server, a test
/// closure). Lemmalog owns the prompt, the response protocol, and
/// memoization by episode id, so an episode is never re-extracted.
///
/// The model is asked to answer in the line protocol `S --rel--> O`
/// (optionally `S --rel[0.8]--> O`). Extraction failures degrade to zero
/// facts rather than poisoning memory.
pub struct LlmExtractor {
    call: Box<dyn FnMut(&str) -> Result<String, String>>,
    default_confidence: f64,
    seen: HashMap<String, Vec<CandidateFact>>,
    pub calls: usize, // observability for tests/metrics
}

pub const EXTRACTION_PROMPT: &str = "\
Extract the factual triples from the episode below. Answer with one triple \
per line in exactly this format, nothing else:\n\
SUBJECT --RELATION[CONFIDENCE]--> OBJECT\n\
CONFIDENCE is a number in [0,1] (omit [CONFIDENCE] for 0.9). RELATION must \
be one of: works_at, manager, likes, job_title, member_of, located_in, \
links. Employment (works at, joined, was hired by, left) is ALWAYS \
works_at - for a job change emit only the NEW employer as a works_at \
triple. Reporting lines (reports to, manager is) are ALWAYS manager, with \
the person as the subject. Use the closest match for anything else; skip \
facts that fit none. SUBJECT and OBJECT must be real entity names exactly \
as written in the episode: NEVER a pronoun or a role word (speaker, user, \
the manager) - always the full name. Output ONLY the triple lines: no \
reasoning, no explanations, no bullets, no questions. Skip opinions and \
small talk.\n\
Episode:\n";

impl LlmExtractor {
    pub fn new<F>(call: F) -> Self
    where
        F: FnMut(&str) -> Result<String, String> + 'static,
    {
        LlmExtractor {
            call: Box::new(call),
            default_confidence: 0.9,
            seen: HashMap::new(),
            calls: 0,
        }
    }
}

impl Extractor for LlmExtractor {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        if let Some(cached) = self.seen.get(&episode.id) {
            return cached.clone();
        }
        self.calls += 1;
        let prompt = format!("{EXTRACTION_PROMPT}{}", episode.text);
        let out = match (self.call)(&prompt) {
            Ok(response) => parse_protocol(&response, self.default_confidence),
            Err(_) => Vec::new(), // degraded turn: no facts, no poison
        };
        self.seen.insert(episode.id.clone(), out.clone());
        out
    }

    fn stats(&self) -> (usize, usize) {
        (self.calls, 0)
    }
}

/// Outcome of one `observe()` — the agent-visible update report.
#[derive(Debug, Default, Clone)]
pub struct IngestReport {
    pub added: usize,
    pub updated: usize,
    pub noop: usize,
    pub escalations: Vec<String>,
}

/// Agent memory facade: engine + extraction + episodes + escalations.
pub struct AgentMemory<X: Extractor> {
    pub engine: Engine,
    extractor: X,
    episodes: Vec<Episode>,
    escalations: Vec<String>,
    episode_counter: u64,
    /// Epoch of the last completed `maintain()`; `context()` reports
    /// memory changes since then.
    last_turn_epoch: u64,
    extra_rules: String,
    hyp_counter: u64,
}

pub const DEFAULT_RULES: &str = "\
# temporal projection: what is true NOW
current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.
# curated exclusivity table for the update policy
exclusive(\"works_at\").
";

impl<X: Extractor> AgentMemory<X> {
    pub fn new(extractor: X, extra_rules: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut engine = Engine::new();
        engine.install_program(DEFAULT_RULES)?;
        if !extra_rules.trim().is_empty() {
            engine.install_program(extra_rules)?;
        }
        Ok(AgentMemory {
            engine,
            extractor,
            episodes: Vec::new(),
            escalations: Vec::new(),
            episode_counter: 0,
            last_turn_epoch: 0,
            extra_rules: extra_rules.to_string(),
            hyp_counter: 0,
        })
    }

    /// Ingest one episode at the current engine time.
    pub fn observe(&mut self, text: &str) -> IngestReport {
        let ts = self.engine.now;
        self.observe_at(text, ts)
    }

    /// Ingest an episode with a known speaker: first-person references
    /// ("I", "my") resolve to `speaker` during extraction.
    pub fn observe_as(&mut self, text: &str, ts: i64, speaker: &str) -> IngestReport {
        self.engine.set_now(ts);
        self.episode_counter += 1;
        let episode = Episode {
            id: format!("ep{}", self.episode_counter),
            text: text.to_string(),
            ts,
            speaker: Some(speaker.to_string()),
        };
        let candidates = self.extractor.extract(&episode);
        let mut report = IngestReport::default();
        for c in &candidates {
            self.apply_update(c, &episode, &mut report);
        }
        self.episodes.push(episode);
        self.escalations.extend(report.escalations.clone());
        report
    }

    /// Ingest one episode at an explicit timestamp: the extraction boundary
    /// is bi-temporal — `ts` becomes both valid-from of new facts and the
    /// closing valid-to of facts they supersede. Call `maintain()` (the
    /// sleep-time slot) afterwards to re-derive.
    pub fn observe_at(&mut self, text: &str, ts: i64) -> IngestReport {
        self.engine.set_now(ts);
        self.episode_counter += 1;
        let episode = Episode {
            id: format!("ep{}", self.episode_counter),
            text: text.to_string(),
            ts,
            speaker: None,
        };
        let candidates = self.extractor.extract(&episode);
        let mut report = IngestReport::default();
        for c in &candidates {
            self.apply_update(c, &episode, &mut report);
        }
        self.episodes.push(episode);
        self.escalations.extend(report.escalations.clone());
        report
    }

    /// Deterministic update decision for one candidate fact:
    /// - no open fact with same (S,P)      -> ADD
    /// - open fact with same (S,P,O)       -> NOOP (annotation merge)
    /// - open fact with different O:
    ///     - P exclusive                   -> UPDATE (close old, assert new)
    ///     - otherwise                     -> ADD + escalation
    fn apply_update(&mut self, c: &CandidateFact, ep: &Episode, report: &mut IngestReport) {
        let subj = self.engine.sym(&c.subj);
        let pred = self.engine.sym(&c.pred);
        // digit-only objects become integers: quantities and amounts
        // asserted through the line protocol must feed `sum`/`count`
        // aggregates and `<`/`>` comparisons, which symbols cannot.
        // Mixed-form tokens (dates "2023-05-14", "$50", ids with dashes)
        // stay symbols.
        let obj = match c.obj.parse::<i64>() {
            Ok(n) => Value::Int(n),
            Err(_) => self.engine.sym(&c.obj),
        };
        let open: Vec<Vec<Value>> = self
            .engine
            .query("edge", &[Some(subj), Some(pred), None, None, None, None])
            .into_iter()
            .map(|(k, _)| k)
            .filter(|k| matches!(k[4].as_int(), Some(vt) if vt == i64::MAX))
            .collect();
        if open.is_empty() {
            self.assert_open(&[subj, pred, obj], c.confidence, &ep.id);
            report.added += 1;
            return;
        }
        if open.iter().any(|k| k[2] == obj) {
            // same fact re-observed: merge annotation, no structural change
            let mut k = open[0].clone();
            k[2] = obj;
            self.engine.declare(
                "edge",
                &k,
                Ann::base(c.confidence, [ep.id.clone()]),
            );
            report.noop += 1;
            return;
        }
        let exclusive = !self.engine.query("exclusive", &[Some(pred)]).is_empty();
        if exclusive {
            for old in &open {
                let mut closed = old.clone();
                closed[4] = Value::Int(self.engine.now);
                self.engine.retract("edge", old);
                self.engine.declare("edge", &closed, Ann::base(0.9, ["superseded"]));
            }
            self.assert_open(&[subj, pred, obj], c.confidence, &ep.id);
            report.updated += 1;
        } else {
            self.assert_open(&[subj, pred, obj], c.confidence, &ep.id);
            let others: Vec<String> = open
                .iter()
                .map(|k| self.engine.interner.display(&k[2]))
                .collect();
            report.escalations.push(format!(
                "conflict: {} --{}--> {} asserted in {}, but {} also open ({})",
                c.subj, c.pred, c.obj, ep.id, c.pred, others.join(", ")
            ));
            report.added += 1;
        }
    }

    fn assert_open(&mut self, spo: &[Value; 3], conf: f64, prov: &str) {
        let args = vec![
            spo[0],
            spo[1],
            spo[2],
            Value::Int(self.engine.now),
            Value::Int(i64::MAX),
            Value::Int(self.engine.now),
        ];
        self.engine.declare("edge", &args, Ann::base(conf, [prov]));
    }

    /// Advance time and run incremental maintenance (the sleep-time slot).
    /// The epoch the run logs under is remembered so the next `context()`
    /// can report this turn's changes ("what's new in memory").
    pub fn maintain(&mut self, now: i64) -> usize {
        self.engine.set_now(now);
        self.last_turn_epoch = self.engine.epoch();
        self.engine.run()
    }

    pub fn escalations(&self) -> &[String] {
        &self.escalations
    }

    /// Dismiss an escalation (agent resolved it out-of-band).
    pub fn resolve_escalation(&mut self, idx: usize) {
        if idx < self.escalations.len() {
            self.escalations.remove(idx);
        }
    }

    /// Agent-facing read-only query: bindings for an atom like
    /// `current("alice", R, O)` against materialized relations.
    pub fn ask(&self, goal: &str) -> Result<Vec<String>, crate::ast::ParseError> {
        self.engine.ask(goal)
    }

    /// Ingest PRE-PARSED facts (callers that extract themselves, e.g. the
    /// MCP server where the host model does extraction): applies the same
    /// update policy as `observe_at`, and returns the drop report for any
    /// lines the caller's protocol parse rejected.
    pub fn observe_extracted(
        &mut self,
        text: &str,
        ts: i64,
    ) -> (IngestReport, Vec<(String, String)>) {
        self.engine.set_now(ts);
        let (candidates, dropped) = parse_protocol_reported(text, 0.9);
        self.episode_counter += 1;
        let episode = Episode {
            id: format!("ep{}", self.episode_counter),
            text: text.to_string(),
            ts,
            speaker: None,
        };
        let mut report = IngestReport::default();
        for c in &candidates {
            self.apply_update(c, &episode, &mut report);
        }
        self.episodes.push(episode);
        self.escalations.extend(report.escalations.clone());
        (report, dropped)
    }

    /// Agent tool surface: install a rule batch (versioned, revertable).
    pub fn install_rules(&mut self, src: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.engine.install_program(src)
    }

    /// Agent tool surface: uninstall a rule batch; derivations revert on
    /// the next `maintain()`.
    pub fn uninstall_rules(&mut self, id: &str) -> bool {
        self.engine.uninstall(id)
    }

    pub fn rule_batches(&self) -> Vec<(String, String)> {
        self.engine.batches()
    }

    /// Lookahead: "what would follow if this episode were true?" Extracts
    /// the episode's candidates (memoized under a hypothetical id, never
    /// colliding with real episodes), evaluates the goal under those
    /// temporary facts, and restores the memory untouched. Returns the
    /// goal bindings and the number of facts the assumption would add.
    pub fn what_if(
        &mut self,
        text: &str,
        goal: &str,
    ) -> Result<(Vec<String>, usize), Box<dyn std::error::Error>> {
        self.hyp_counter += 1;
        let episode = Episode {
            id: format!("hyp{}", self.hyp_counter),
            text: text.to_string(),
            ts: self.engine.now,
            speaker: None,
        };
        let candidates = self.extractor.extract(&episode);
        let now = self.engine.now;
        let extras: Vec<(String, Vec<Value>)> = candidates
            .iter()
            .map(|c| {
                let obj = match c.obj.parse::<i64>() {
                    Ok(n) => Value::Int(n),
                    Err(_) => self.engine.sym(&c.obj),
                };
                (
                    "edge".to_string(),
                    vec![
                        self.engine.sym(&c.subj),
                        self.engine.sym(&c.pred),
                        obj,
                        Value::Int(now),
                        Value::Int(i64::MAX),
                        Value::Int(now),
                    ],
                )
            })
            .collect();
        let refs: Vec<(&str, &[Value])> = extras
            .iter()
            .map(|(p, a)| (p.as_str(), a.as_slice()))
            .collect();
        let rows = self.engine.hypothetical(&refs, goal)?;
        Ok((rows, self.engine.last_hypothetical_facts))
    }

    /// Query `near` relevance facts for a session/entity pair.
    pub fn query_near(
        &self,
        session: crate::intern::Value,
        entity: crate::intern::Value,
    ) -> Vec<(Vec<crate::intern::Value>, crate::eval::Ann)> {
        self.engine.query("near", &[Some(session), Some(entity), None])
    }

    /// Demand-driven query (magic sets): answers without materializing the
    /// full fixpoint of the queried predicate. Runs an (idle-cheap)
    /// maintenance pass first so all-free adornments can alias fresh
    /// materialized relations instead of re-deriving closures.
    pub fn ask_deep(&mut self, goal: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let now = self.engine.now;
        self.maintain(now);
        self.engine.ask_deep(goal)
    }

    pub fn why(&self, fact: &str) -> String {
        match crate::ast::parse_program(&format!("{fact}.")) {
            Ok(clauses) if clauses.len() == 1 => {
                let head = &clauses[0].head;
                match self.engine.ground_values(&head.args) {
                    Some(args) => self.engine.why(&head.pred, &args),
                    None => format!("why: {fact} contains variables"),
                }
            }
            _ => format!("why: cannot parse {fact:?}"),
        }
    }

    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }

    /// Extractor observability: (model calls, failures).
    pub fn extractor_stats(&self) -> (usize, usize) {
        self.extractor.stats()
    }

    /// Assemble the context window for a query mentioning `entities`:
    /// distilled facts first, verbatim provenance last, under budget. A
    /// leading "what changed in memory" section reports facts created since
    /// the last `maintain()` (capped).
    pub fn context(&self, entities: &[&str], budget_tokens: usize) -> String {
        let news: Vec<String> = self
            .engine
            .changes_from(self.last_turn_epoch)
            .iter()
            .take(20)
            .map(|(p, a)| self.engine.render_fact(p, a))
            .collect();
        assemble_context(&self.engine, &self.episodes, entities, budget_tokens, &news)
    }

    /// Query-driven context assembly via hybrid retrieval: BM25 over facts
    /// and episodes + entity-match boosting (a query naming an entity pulls
    /// that entity's facts and one-hop neighbors), budget-aware, distilled
    /// facts first and their provenance episodes last. This is the
    /// "selection, not extraction" answer to context bloat.
    pub fn context_for_query(&self, query: &str, budget_tokens: usize) -> String {
        let r = crate::retrieval::Retrieval::build(&self.engine, &self.episodes);
        let sel = r.select(query, budget_tokens);
        r.render(&sel)
    }

    /// Retract base facts given in the line protocol. Each open `edge`
    /// row matching (S, R, O) is removed; `maintain()` then recomputes
    /// the transitive dependents (scoped negative delta). Returns
    /// (retracted lines, not-found lines, derived facts that died) —
    /// the consequence report is the point: the caller sees exactly
    /// what invalidation propagated.
    pub fn retract_facts(
        &mut self,
        text: &str,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        let candidates = parse_protocol_strict(text, 0.9);
        let mut done = Vec::new();
        let mut missing = Vec::new();
        for c in &candidates {
            let subj = self.engine.sym(&c.subj);
            let pred = self.engine.sym(&c.pred);
            let obj = match c.obj.parse::<i64>() {
                Ok(n) => Value::Int(n),
                Err(_) => self.engine.sym(&c.obj),
            };
            let open: Vec<Vec<Value>> = self
                .engine
                .query("edge", &[Some(subj), Some(pred), Some(obj), None, None, None])
                .into_iter()
                .map(|(k, _)| k)
                .filter(|k| matches!(k[4].as_int(), Some(vt) if vt == i64::MAX))
                .collect();
            if open.is_empty() {
                missing.push(format!("{} --{}--> {}", c.subj, c.pred, c.obj));
                continue;
            }
            for row in &open {
                self.engine.retract("edge", row);
            }
            done.push(format!("{} --{}--> {}", c.subj, c.pred, c.obj));
        }
        if done.is_empty() {
            return (done, missing, Vec::new());
        }
        // consequence report: snapshot derived relations before the
        // recompute, diff after — scoped recompute logs predicate-level
        // clears, so per-row feed events don't tell the story
        let derived_preds: Vec<String> = self.engine.ever_derived.iter().cloned().collect();
        let mut before: std::collections::BTreeMap<String, std::collections::BTreeSet<Vec<Value>>> =
            std::collections::BTreeMap::new();
        for p in &derived_preds {
            before.insert(
                p.clone(),
                self.engine.relation_keys(p).into_iter().collect(),
            );
        }
        let now = self.engine.now;
        self.maintain(now);
        let mut died = Vec::new();
        for (p, old_keys) in &before {
            let now_keys: std::collections::BTreeSet<Vec<Value>> =
                self.engine.relation_keys(p).into_iter().collect();
            for k in old_keys.difference(&now_keys) {
                died.push(self.engine.render_fact(p, k));
            }
        }
        (done, missing, died)
    }

    /// `context_for_query` plus the misattribution and supersession
    /// sections: an ATTRIBUTION contrast (which subjects hold facts on
    /// the question's topic — zero counts are the false-premise signal)
    /// and, for current-state/value questions, the newest value per
    /// slot with supersessions listed as history. Same facts, but the
    /// reader cannot confuse Caroline's gift with Melanie's.
    pub fn context_for_query_rich(&self, query: &str, budget_tokens: usize) -> String {
        let mut base = self.context_for_query(query, budget_tokens);
        let qt = crate::retrieval::tokens3(query);
        // attribution contrast
        {
            let mut holders: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            let mut subjects: Vec<String> = Vec::new();
            for key in self.engine.relation_keys("current") {
                if key.len() != 3 {
                    continue;
                }
                let subj = self.engine.interner.display(&key[0]).to_string();
                if !subjects.contains(&subj) {
                    subjects.push(subj.clone());
                }
                let line = format!(
                    "{} --{}--> {}",
                    subj,
                    self.engine.interner.display(&key[1]),
                    self.engine.interner.display(&key[2])
                );
                if crate::retrieval::topic_overlap(&crate::retrieval::tokens3(&line), &subj, &qt)
                    >= 1
                {
                    *holders.entry(subj).or_insert(0) += 1;
                }
            }
            let asked: Vec<String> = subjects
                .iter()
                .filter(|s| {
                    crate::retrieval::tokens3(s)
                        .iter()
                        .any(|t| t.len() >= 3 && qt.contains(t))
                })
                .cloned()
                .collect();
            let zero: Vec<String> = asked
                .iter()
                .filter(|s| !holders.contains_key(*s))
                .cloned()
                .collect();
            if !holders.is_empty() || !zero.is_empty() {
                let mut sec = String::from(
                    "\nATTRIBUTION (who holds facts on this question's topic, by subject):\n",
                );
                let mut pairs: Vec<_> = holders.iter().collect();
                pairs.sort_by(|a, b| b.1.cmp(a.1));
                for (s, n) in pairs.iter().take(6) {
                    sec.push_str(&format!("  {s}: {n} topic facts\n"));
                }
                if !zero.is_empty() {
                    sec.push_str(&format!(
                        "  NO topic facts for: {} (mentioned in the question)\n",
                        zero.join(", ")
                    ));
                }
                base.push_str(&sec);
            }
        }
        // latest values per slot (supersession history)
        let q_lower = query.to_lowercase();
        if q_lower.contains("current")
            || q_lower.contains(" now")
            || q_lower.contains("latest")
            || q_lower.contains("amount")
            || q_lower.contains("price")
            || q_lower.contains("how much")
            || q_lower.contains("value")
        {
            let mut slots: std::collections::BTreeMap<(String, String), Vec<(i64, String)>> =
                std::collections::BTreeMap::new();
            for key in self.engine.relation_keys("edge") {
                if key.len() != 6 {
                    continue;
                }
                let subj = self.engine.interner.display(&key[0]).to_string();
                let rel = self.engine.interner.display(&key[1]).to_string();
                let line = format!("{subj} --{rel}--> {}", self.engine.interner.display(&key[2]));
                if !crate::retrieval::tokens3(&line)
                    .iter()
                    .any(|t| t.len() >= 3 && qt.contains(t))
                {
                    continue;
                }
                let vf = key[3].as_int().unwrap_or(0);
                if vf == i64::MAX {
                    continue;
                }
                slots
                    .entry((subj, rel))
                    .or_default()
                    .push((vf, self.engine.interner.display(&key[2]).to_string()));
            }
            let mut sec = String::from(
                "\nCURRENT STATE (latest value per slot; superseded values listed as history):\n",
            );
            let mut lines = 0;
            for ((subj, rel), mut vals) in slots {
                if lines >= 8 {
                    break;
                }
                vals.sort_by(|a, b| b.0.cmp(&a.0));
                let distinct: Vec<String> = vals
                    .iter()
                    .map(|(_, v)| v.clone())
                    .fold(Vec::new(), |mut acc: Vec<String>, v| {
                        if !acc.contains(&v) {
                            acc.push(v);
                        }
                        acc
                    });
                if distinct.len() < 2 {
                    continue;
                }
                let older: Vec<String> = distinct.iter().skip(1).take(3).cloned().collect();
                sec.push_str(&format!(
                    "  {subj} --{rel}--> {} (current; superseded: {})\n",
                    distinct[0],
                    older.join(", ")
                ));
                lines += 1;
            }
            if lines > 0 {
                base.push_str(&sec);
            }
        }
        base
    }
}

// ------------------------------------------------------ context assembler

/// Positional assembly (lost-in-the-middle mitigation): derived high-value
/// facts at the top of the window, verbatim source episodes at the bottom,
/// byte budget `tokens * 4` split 60/40.
pub fn assemble_context(
    engine: &Engine,
    episodes: &[Episode],
    entities: &[&str],
    budget_tokens: usize,
    news: &[String],
) -> String {
    let mut relevant: Vec<(Vec<Value>, Ann)> = Vec::new();
    for name in entities {
        let v = engine.sym_of(name);
        relevant.extend(engine.query("current", &[Some(v), None, None]));
    }
    relevant.sort_by(|a, b| b.1.conf.partial_cmp(&a.1.conf).unwrap_or(std::cmp::Ordering::Equal));

    let distilled_budget = (budget_tokens * 4 * 6 / 10).max(0);
    let mut distilled = String::new();
    let mut used_prov: Vec<String> = Vec::new();
    for (k, ann) in &relevant {
        let line = format!(
            "{} --{}--> {}   [conf {:.2}, prov {}]\n",
            engine.interner.display(&k[0]),
            engine.interner.display(&k[1]),
            engine.interner.display(&k[2]),
            ann.conf,
            ann.prov.iter().cloned().collect::<Vec<_>>().join(",")
        );
        if distilled.len() + line.len() > distilled_budget {
            break;
        }
        distilled.push_str(&line);
        used_prov.extend(ann.prov.iter().cloned());
    }

    let source_budget = (budget_tokens * 4 * 4 / 10).max(0);
    let mut sources = String::new();
    let mut used: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for ep in episodes {
        if !used_prov.iter().any(|p| p == &ep.id) {
            continue;
        }
        used.insert(&ep.id);
        let block = format!("[{}] {}\n", ep.id, ep.text);
        if sources.len() + block.len() > source_budget {
            break;
        }
        sources.push_str(&block);
    }
    let _ = used;

    let mut out = String::new();
    if !news.is_empty() {
        let _ = writeln!(out, "== new in memory since last turn ==");
        for line in news {
            let _ = writeln!(out, "{line}");
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out, "== memory (distilled, highest confidence first) ==");
    out.push_str(&distilled);
    let _ = writeln!(out, "\n== source episodes (verbatim, from provenance) ==");
    out.push_str(&sources);
    out
}

impl Engine {
    /// Non-mutating symbol lookup for read-only paths.
    pub fn sym_of(&self, s: &str) -> Value {
        match self.interner.lookup(s) {
            Some(v) => Value::Sym(v),
            None => Value::Int(i64::MIN), // never matches: unknown entity
        }
    }

    /// Resolve a pattern to ground values for `why()` (None if vars remain).
    pub fn ground_values(&self, pat: &[Term]) -> Option<Vec<Value>> {
        pat.iter()
            .map(|t| match t {
                Term::Sym(s) => self.interner.lookup(s).map(Value::Sym),
                Term::Int(i) => Some(Value::Int(*i)),
                _ => None,
            })
            .collect()
    }
}

// ------------------------------------------------------------- persistence

/// Argument representation while parsing a snapshot (symbols intern
/// against the target engine once it exists).
enum ArgRepr {
    S(String),
    I(i64),
}

/// Escape a field for the tab-separated snapshot format.
fn esc(s: &str) -> String {
    // spaces too: snapshot fields and fact args are space-separated, so
    // multi-word symbols (extracted entities like "United Airlines") must
    // not split on reload
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace(' ', "\\s")
}

fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('s') => out.push(' '),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

const SNAPSHOT_MAGIC: &str = "LEMMALOG1";
/// The batch `new()` installs from `DEFAULT_RULES`; never persisted.
const BOOTSTRAP_BATCH: &str = "b0";
/// Pre-rename snapshots (read-only compatibility).
const SNAPSHOT_MAGIC_V0: &str = "CORTEXLOG1";

impl<X: Extractor> AgentMemory<X> {
    /// Persist to a snapshot file: rules, clock, episodes (verbatim
    /// sources), escalation queue, and all base (EDB) facts with their
    /// annotations. Derived relations are NOT persisted — they are
    /// rebuildable projections, recomputed by `load()`.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "{SNAPSHOT_MAGIC}");
        let _ = writeln!(out, "NOW\t{}", self.engine.now);
        let _ = writeln!(out, "RULES\t{}", esc(&self.extra_rules));
        // Batches installed after construction (`install_rules`) are not in
        // `extra_rules`, so persisting only that field drops every rule an
        // agent installed — silently, since base facts still load and the
        // derived views simply come back empty. Batch ids are positional
        // (`b{len}`), so replaying non-bootstrap batches in order restores
        // them under the same ids and `uninstall` keeps working.
        for (id, src) in self.engine.batches() {
            if id == BOOTSTRAP_BATCH {
                continue; // reinstalled by `new()` from DEFAULT_RULES
            }
            let _ = writeln!(out, "RULEB\t{}\t{}", esc(&id), esc(&src));
        }
        for ep in &self.episodes {
            let _ = writeln!(
                out,
                "EP\t{}\t{}\t{}\t{}",
                esc(&ep.id),
                ep.ts,
                ep.speaker.as_deref().unwrap_or(""),
                esc(&ep.text)
            );
        }
        for e in &self.escalations {
            let _ = writeln!(out, "ESC\t{}", esc(e));
        }
        for (pred, rel) in &self.engine.relations {
            // base facts only: skip predicates defined by rules (they are
            // either program facts re-declared from rules, or derived)
            if self.engine.clauses.iter().any(|c| c.head.pred == *pred) {
                continue;
            }
            for row in &rel.rows {
                let prov = row.fact.ann.prov.iter().cloned().collect::<Vec<_>>().join(",");
                let args = row
                    .key
                    .iter()
                    .map(|v| match v {
                        Value::Sym(s) => format!("s:{}", esc(self.engine.interner.resolve(*s))),
                        Value::Int(i) => format!("i:{i}"),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let _ = writeln!(
                    out,
                    "FACT\t{}\t{}\t{}\t{}",
                    pred, row.fact.ann.conf, prov, args
                );
            }
        }
        std::fs::write(path, out)
    }

    /// Load a snapshot into a fresh memory with the given extractor.
    /// Base facts are re-asserted with their annotations; derived
    /// relations are rebuilt by one maintenance run.
    pub fn load(
        extractor: X,
        path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let magic = lines.next();
        if magic != Some(SNAPSHOT_MAGIC) && magic != Some(SNAPSHOT_MAGIC_V0) {
            return Err("not a lemmalog snapshot".into());
        }
        let mut rules = String::new();
        let mut batch_srcs: Vec<String> = Vec::new();
        let mut now = 0i64;
        let mut episodes = Vec::new();
        let mut escalations = Vec::new();
        let mut facts: Vec<(String, f64, Vec<String>, Vec<ArgRepr>)> = Vec::new();
        for line in lines {
            let Some((tag, rest)) = line.split_once('\t') else {
                continue;
            };
            match tag {
                "NOW" => now = rest.parse()?,
                "RULES" => rules = unesc(rest),
                "RULEB" => {
                    let mut f = rest.splitn(2, '\t');
                    let (Some(_id), Some(src)) = (f.next(), f.next()) else {
                        return Err("bad RULEB record".into());
                    };
                    batch_srcs.push(unesc(src));
                }
                "EP" => {
                    let mut f = rest.splitn(4, '\t');
                    let (Some(id), Some(ts), Some(speaker), Some(txt)) =
                        (f.next(), f.next(), f.next(), f.next())
                    else {
                        return Err("bad EP record".into());
                    };
                    episodes.push(Episode {
                        id: unesc(id),
                        ts: ts.parse()?,
                        speaker: if speaker.is_empty() {
                            None
                        } else {
                            Some(unesc(speaker))
                        },
                        text: unesc(txt),
                    });
                }
                "ESC" => escalations.push(unesc(rest)),
                "FACT" => {
                    let mut f = rest.splitn(4, '\t');
                    let (Some(pred), Some(conf), Some(prov), Some(args)) =
                        (f.next(), f.next(), f.next(), f.next())
                    else {
                        return Err("bad FACT record".into());
                    };
                    let mut vals = Vec::new();
                    for a in args.split(' ').filter(|s| !s.is_empty()) {
                        if let Some(sym) = a.strip_prefix("s:") {
                            vals.push(ArgRepr::S(unesc(sym)));
                        } else if let Some(i) = a.strip_prefix("i:") {
                            vals.push(ArgRepr::I(i.parse()?));
                        } else {
                            return Err(format!("bad arg {a:?}").into());
                        }
                    }
                    let prov: Vec<String> = prov
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(unesc)
                        .collect();
                    facts.push((pred.to_string(), conf.parse()?, prov, vals));
                }
                _ => {}
            }
        }
        let mut m = if batch_srcs.is_empty() {
            // Legacy snapshot (or nothing installed past the bootstrap).
            AgentMemory::new(extractor, &rules)?
        } else {
            // `extra_rules` became a batch at construction, so it is already
            // in `batch_srcs`; passing it again would install it twice.
            let mut m = AgentMemory::new(extractor, "")?;
            for src in &batch_srcs {
                m.engine.install_program(src)?;
            }
            m.extra_rules = rules;
            m
        };
        m.escalations = escalations;
        m.episodes = episodes;
        m.episode_counter = m.episodes.len() as u64;
        for (pred, conf, prov, args) in facts {
            let resolved: Vec<Value> = args
                .into_iter()
                .map(|v| match v {
                    ArgRepr::S(name) => m.engine.sym(&name),
                    ArgRepr::I(i) => Value::Int(i),
                })
                .collect();
            m.engine.declare(&pred, &resolved, Ann::base(conf, prov));
        }
        m.engine.set_now(now);
        let _ = m.engine.run();
        m.last_turn_epoch = m.engine.epoch();
        Ok(m)
    }
}
