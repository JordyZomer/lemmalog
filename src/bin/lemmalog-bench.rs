//! lemmalog-bench: the bridge binary for the MemEval harness.
//!
//! Two commands:
//!   lemmalog-bench ingest  <conv.json> <snapshot-path>   # extraction (Claude,
//!     chunked, file-cached) + update policy + derived rules + date
//!     normalization + entity reconciliation -> snapshot
//!   lemmalog-bench context <snapshot-path> <question>    # hybrid retrieval
//!     (BM25 + graph boosts + embedding rerank) -> assembled memory context
//!   lemmalog-bench recall  <snapshot-path> <question>    # targeted re-read
//!     fallback when the structured store lacks the answer
//!
//! The conv.json format is MemEval's normalized LoCoMo shape (session_N keys
//! with {speaker, text} turns and session_N_date_time) or raw LoCoMo
//! ("1:56 pm on 8 May, 2023" dates).
//!
//! Env: ANTHROPIC_API_KEY (extraction + reconciliation), LEMMALOG_EXTRACT_MODEL
//! (default claude-sonnet-4-6), LEMMALOG_CACHE_DIR (extraction cache: pay
//! once, rerun free), LEMMALOG_CONTEXT_BUDGET (default 1800),
//! LEMMALOG_EMBED_BASE (embedding endpoint for the semantic rerank and
//! reconciliation candidate gating; default local LM Studio nomic, "off"
//! disables).

#![cfg(feature = "llm")]

use lemmalog::agent::{AgentMemory, MockExtractor};
use lemmalog::canonical;
use lemmalog::eval::Engine;
use lemmalog::intern::Value;
use lemmalog::llm::{LlmClientExtractor, OpenAiClient};
use lemmalog::retrieval::{tokenize_pub, Retrieval};
use lemmalog::semantics::Embedder as _;
use serde_json::Value as J;
use std::collections::BTreeMap;
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("ingest") => {
            let (conv_path, snap) = (
                args.get(2).expect("usage: ingest <conv.json> <snapshot>"),
                args.get(3).expect("usage: ingest <conv.json> <snapshot>"),
            );
            ingest(conv_path, snap);
        }
        Some("context") => {
            let (snap, question) = (
                args.get(2).expect("usage: context <snapshot> <question>"),
                args.get(3).expect("usage: context <snapshot> <question>"),
            );
            let ctx = context(snap, question);
            print!("{ctx}");
        }
        Some("hasevidence") => {
            let (snap, question) = (
                args.get(2).expect("usage: hasevidence <snapshot> <question>"),
                args.get(3).expect("usage: hasevidence <snapshot> <question>"),
            );
            let mut m = AgentMemory::load(MockExtractor::new(0.9), snap).expect("load snapshot");
            let _ = m.maintain(m.engine.now);
            for l in evidence_lines(&m, snap, question) {
                println!("{l}");
            }
        }
        Some("recall") => {
            let (snap, question) = (
                args.get(2).expect("usage: recall <snapshot> <question>"),
                args.get(3).expect("usage: recall <snapshot> <question>"),
            );
            let recalled = recall(snap, question);
            print!("{recalled}");
        }
        _ => {
            eprintln!("usage: lemmalog-bench ingest|context|recall ...");
            std::process::exit(2);
        }
    }
}

/// "2023/04/10 (Mon) 17:50" or LoCoMo's "1:56 pm on 8 May, 2023" ->
/// monotonic minutes (matches the runner).
fn parse_date(s: &str) -> i64 {
    let parts: Vec<&str> = s.split_whitespace().collect();
    // normalized MemEval shape: date first, time at index 3 ("... 13:39 GMT")
    // or 2 ("2023/04/27 (Thu) 13:39")
    if let Some(d) = parts.first() {
        if d.contains('/') {
            let t = parts
                .get(3)
                .or_else(|| parts.get(2))
                .copied()
                .unwrap_or("0:0");
            let dp: Vec<i64> = d.split('/').filter_map(|x| x.parse().ok()).collect();
            let tp: Vec<i64> = t.split(':').filter_map(|x| x.parse().ok()).collect();
            return (((dp.first().copied().unwrap_or(0) * 366
                + dp.get(1).copied().unwrap_or(0) * 31
                + dp.get(2).copied().unwrap_or(0))
                * 24
                + tp.first().copied().unwrap_or(0))
                * 60)
                + tp.get(1).copied().unwrap_or(0);
        }
    }
    // raw LoCoMo: "1:56 pm on 8 May, 2023"
    const MONTHS: [&str; 12] = [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    ];
    let lower = s.to_lowercase();
    if let Some(pos) = lower.find(" on ") {
        let date_part = &lower[pos + 4..];
        let toks: Vec<&str> = date_part.split_whitespace().collect();
        if toks.len() >= 3 {
            let day: i64 = toks[0].trim_end_matches(',').parse().unwrap_or(0);
            let mon = MONTHS
                .iter()
                .position(|m| toks[1].trim_end_matches(',').starts_with(m))
                .unwrap_or(0) as i64
                + 1;
            let year: i64 = toks[2].trim_end_matches(',').parse().unwrap_or(0);
            return (year * 366 + mon * 31 + day) * 24 * 60;
        }
    }
    0
}

const RULES: &str = "\
current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.\n\
reports_to(X,Y) :- current(X,\"manager\",Y).\n\
trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).\n\
first_dated(A, min(D)) :- dated(A, D).\n\
happened_before(A, B) :- first_dated(A, D1), first_dated(B, D2), A \\= B, D1 < D2.\n\
stated_before(X, Y) :- current(X, \"before\", Y).\n\
stated_before(X, Z) :- stated_before(X, Y), stated_before(Y, Z).\n";

fn client() -> OpenAiClient {
    let model = std::env::var("LEMMALOG_EXTRACT_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-6".to_string());
    OpenAiClient::new("https://api.anthropic.com/v1", &model, "")
}

fn embed_base() -> Option<String> {
    let b = std::env::var("LEMMALOG_EMBED_BASE")
        .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());
    if b.is_empty() || b == "off" {
        None
    } else {
        Some(b)
    }
}

fn ingest(conv_path: &str, snap: &str) {
    let mut raw = String::new();
    std::fs::File::open(conv_path)
        .expect("open conv.json")
        .read_to_string(&mut raw)
        .expect("read conv.json");
    let conv: J = serde_json::from_str(&raw).expect("parse conv.json");

    // collect (date, text) sessions in numeric order
    let mut sessions: BTreeMap<usize, (String, String)> = BTreeMap::new();
    if let Some(c) = conv.get("conversation").and_then(|c| c.as_object()) {
        for (k, v) in c {
            if let Some(num) = k
                .strip_prefix("session_")
                .and_then(|n| n.parse::<usize>().ok())
            {
                let date = c
                    .get(&format!("session_{num}_date_time"))
                    .and_then(|d| d.as_str())
                    .unwrap_or_default()
                    .to_string();
                let mut text = format!("Session {num} on {date}:\n");
                if let Some(turns) = v.as_array() {
                    for t in turns {
                        let speaker = t.get("speaker").and_then(|s| s.as_str()).unwrap_or("user");
                        let body = t.get("text").and_then(|s| s.as_str()).unwrap_or_default();
                        text.push_str(&format!("{speaker}: {body}\n"));
                    }
                }
                sessions.insert(num, (date, text));
            }
        }
    }
    eprintln!(
        "lemmalog-bench: {} sessions, {} chars total",
        sessions.len(),
        sessions.values().map(|(_, t)| t.len()).sum::<usize>()
    );

    let base = client();
    let extractor = LlmClientExtractor::new(base).with_prompt(lemmalog::longmemeval::OPEN_VOCAB_PROMPT);
    let extractor: Box<dyn lemmalog::agent::Extractor> = match std::env::var("LEMMALOG_CACHE_DIR") {
        Ok(dir) => Box::new(extractor.file_cached(&dir)),
        Err(_) => Box::new(extractor),
    };
    let mut m = AgentMemory::new(extractor, RULES).expect("memory");

    // sidecar: monotonic ts -> date string, for human-readable history
    let mut dates: BTreeMap<String, String> = BTreeMap::new();
    let mut total_facts = 0usize;
    for (num, (date, text)) in &sessions {
        let ts = parse_date(date);
        dates.insert(ts.to_string(), date.clone());
        let report = m.observe_as(text, ts, "the_user");
        total_facts += report.added + report.updated;
        m.maintain(ts);
        eprintln!("  session {num}: +{} facts", report.added);
    }
    // normalize date-shaped objects to comparable integers: dated/2 feeds
    // happened_before, and the engine's `<` on symbols compares intern ids,
    // not date text — so dates must be Ints to order correctly
    let mut dated_count = 0usize;
    for key in m.engine.relation_keys("current") {
        if key.len() != 3 {
            continue;
        }
        let obj = m.engine.interner.display(&key[2]).to_string();
        if let Some(n) = lemmalog::longmemeval::date_to_int(&obj) {
            let subj = key[0];
            if m
                .engine
                .declare("dated", &[subj, Value::Int(n)], lemmalog::eval::Ann::base(0.95, ["date_norm"]))
            {
                dated_count += 1;
            }
        }
    }
    eprintln!("  temporal: {dated_count} dated facts normalized");
    // plain-number objects (amounts, quantities) -> numeric(S, R, Int) so
    // the aggregation engine can sum them ("which store did I spend the
    // most at" is a per-group sum, not a count)
    let mut numeric_count = 0usize;
    for key in m.engine.relation_keys("current") {
        if key.len() != 3 {
            continue;
        }
        let obj = m.engine.interner.display(&key[2]).to_string();
        let is_plain_number = !obj.is_empty()
            && obj.len() <= 6
            && obj.chars().all(|c| c.is_ascii_digit())
            && lemmalog::longmemeval::date_to_int(&obj).is_none();
        if is_plain_number {
            let n: i64 = obj.parse().unwrap_or(0);
            if m
                .engine
                .declare("numeric", &[key[0], key[1], Value::Int(n)], lemmalog::eval::Ann::base(0.95, ["num_norm"]))
            {
                numeric_count += 1;
            }
        }
    }
    eprintln!("  numeric: {numeric_count} amount facts normalized");
    // entity reconciliation: canonical rules + alias pass (one LLM call)
    m.engine
        .install_program(canonical::CANONICAL_RULES)
        .expect("canonical rules");
    m.engine.seed_entities(&["current"]);
    let aliases = reconcile(&mut m.engine, embed_base().as_deref());
    eprintln!(
        "  reconcile: {} aliases asserted, {} conflicts",
        aliases.len(),
        canonical::alias_conflicts(&m.engine).len()
    );
    let last_ts = sessions
        .values()
        .map(|(d, _)| parse_date(d))
        .max()
        .unwrap_or(0);
    let _ = m.maintain(last_ts);
    m.save(snap).expect("save snapshot");
    let sidecar = format!("{snap}.dates.json");
    std::fs::write(&sidecar, serde_json::to_string(&dates).unwrap()).expect("write dates");
    eprintln!(
        "lemmalog-bench: ingested {total_facts} facts -> {snap} (extractor: {} calls)",
        m.extractor_stats().0
    );
}

/// One entity-reconciliation pass over `current`: collect entity names,
/// build candidate pairs (substring containment, same (subject, relation)
/// co-objects, embedding-similar), confirm with one LLM call, assert
/// star-shaped alias edges. Returns the asserted aliases.
fn reconcile(e: &mut Engine, embed: Option<&str>) -> Vec<(String, String, f64)> {
    let mut names: Vec<String> = Vec::new();
    for key in e.relation_keys("current") {
        if key.len() != 3 {
            continue;
        }
        for pos in [0usize, 2] {
            let n = e.interner.display(&key[pos]).to_string();
            // dates and bare numbers are not entity names to merge
            if n.chars().any(|c| c.is_alphabetic()) && !names.contains(&n) {
                names.push(n);
            }
        }
    }
    if names.len() < 2 {
        return Vec::new();
    }
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut push = |pairs: &mut Vec<(String, String)>, a: &str, b: &str| {
        let (a, b) = (a.to_string(), b.to_string());
        if a == b {
            return;
        }
        let flipped = (a.clone(), b.clone());
        let seen = pairs.iter().any(|(x, y)| {
            (x == &a && y == &b) || (x == &flipped.0 && y == &flipped.1)
        });
        if !seen {
            pairs.push((a, b));
        }
    };
    // 1. substring containment: "civic" in "honda civic"
    for i in 0..names.len() {
        for j in i + 1..names.len() {
            let (la, lb) = (names[i].to_lowercase(), names[j].to_lowercase());
            if la.len() >= 3 && lb.len() >= 3 && (la.contains(&lb) || lb.contains(&la)) {
                push(&mut pairs, &names[i], &names[j]);
            }
        }
    }
    // 2. co-reference slots: same (subject, relation), different objects —
    // "owns(user, honda civic)" vs "owns(user, the car)"
    let mut slots: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for key in e.relation_keys("current") {
        if key.len() != 3 {
            continue;
        }
        let s = e.interner.display(&key[0]).to_string();
        let r = e.interner.display(&key[1]).to_string();
        let o = e.interner.display(&key[2]).to_string();
        if o.chars().any(|c| c.is_alphabetic()) {
            let v = slots.entry((s, r)).or_default();
            if !v.contains(&o) {
                v.push(o);
            }
        }
    }
    for objs in slots.values() {
        if objs.len() > 1 && objs.len() <= 8 {
            for i in 0..objs.len() {
                for j in i + 1..objs.len() {
                    push(&mut pairs, &objs[i], &objs[j]);
                }
            }
        }
    }
    // 3. embedding-similar pairs (local nomic): cosine gate, as in
    // canonical::reconcile — semantic near-duplicates the string passes miss
    if let Some(base) = embed {
        let embedder = lemmalog::llm::HttpEmbedder::new(
            base,
            "text-embedding-nomic-embed-text-v1.5",
        );
        let vecs: Vec<Vec<f32>> = names.iter().map(|n| embedder.embed(n)).collect();
        if !vecs.iter().any(|v| v.is_empty()) {
            for i in 0..names.len() {
                for j in i + 1..names.len() {
                    let cos = lemmalog::semantics::cosine_pub(&vecs[i], &vecs[j]);
                    if cos > 0.72 {
                        push(&mut pairs, &names[i], &names[j]);
                    }
                }
            }
        }
    }
    pairs.truncate(120);
    if pairs.is_empty() {
        return Vec::new();
    }
    let listing = pairs
        .iter()
        .map(|(a, b)| format!("- {a} | {b}"))
        .collect::<Vec<_>>()
        .join("\n");
    let chat = client();
    let Ok(out) = chat.chat(
        canonical::reconcile::RECONCILE_PROMPT,
        &format!("Candidate pairs:\n{listing}\n\nLines only:"),
    ) else {
        return Vec::new();
    };
    let mut asserted = Vec::new();
    for f in lemmalog::agent::parse_protocol_strict(&out, 0.8) {
        if f.pred == "alias_of" {
            canonical::assert_alias(e, &f.subj, &f.obj, f.confidence);
            asserted.push((f.subj, f.obj, f.confidence));
        }
    }
    asserted
}

/// "Counting-shaped" questions get a different context: count aggregates
/// (via the aggregation engine) plus more raw episode text — gpt-4.1
/// counts from prose better than our fact count when extraction misses
/// enumerable items, so give it both.
fn is_counting_question(q: &str) -> bool {
    let l = q.to_lowercase();
    l.contains("how many")
        || l.contains("how much")
        || l.contains("total ")
        || l.contains("in total")
        || l.contains("number of")
        // superlatives are aggregation too: "which store did I spend the
        // most at" needs per-group counts, not a single fact
        || l.contains(" most ")
        || l.contains(" least ")
}

/// Lenient plural stem for relevance matching: norm_token only folds
/// plurals longer than 4 chars, so "owns" never met "own" and count
/// lines were dropped exactly when they mattered.
fn stem3(t: &str) -> String {
    if t.len() >= 3 && t.ends_with('s') && !t.ends_with("ss") {
        t[..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

fn tokens3(s: &str) -> std::collections::BTreeSet<String> {
    tokenize_pub(s).into_iter().map(|t| stem3(&t)).collect()
}

/// Topic overlap between a fact row and a question, EXCLUDING the
/// subject's own name tokens: a fact "Melanie --likes--> hiking" overlaps
/// a question about Melanie only through its relation/object. This is
/// what makes misattribution detectable — "grandma's gift to Melanie"
/// scores 0 on Melanie's facts (the gift story is Caroline's).
fn topic_overlap(line_tokens: &std::collections::BTreeSet<String>, subj: &str, qt: &std::collections::BTreeSet<String>) -> usize {
    let subj_toks = tokens3(subj);
    line_tokens
        .iter()
        .filter(|t| {
            t.len() >= 3
                && !subj_toks.contains(*t)
                && qt.iter().any(|q| {
                    q == *t || (q.len() >= 4 && t.len() >= 4 && (q.starts_with(t.as_str()) || t.starts_with(q.as_str())))
                })
        })
        .count()
}

/// Facts whose relation/object content matches the question (excluding
/// the subject name). Backs the refusal-retry: a retry is only offered
/// when the store genuinely holds topic evidence. Lexical overlap OR
/// embedding cosine — "kitchen gadget" has no lexical bridge to
/// "Instant Pot", but the semantic half finds it.
fn evidence_lines(m: &AgentMemory<MockExtractor>, snap: &str, question: &str) -> Vec<String> {
    let qt = tokens3(question);
    let mut scored: Vec<(f64, String)> = Vec::new();
    // semantic half: reuse the per-snapshot embedding cache if present
    let embeds: std::collections::HashMap<String, Vec<f32>> = std::fs::read_to_string(format!("{snap}.embed.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let qvec = embed_base().map(|b| {
        let e = lemmalog::llm::HttpEmbedder::new(&b, "text-embedding-nomic-embed-text-v1.5");
        e.embed(question)
    });
    for key in m.engine.relation_keys("current") {
        if key.len() != 3 {
            continue;
        }
        let subj = m.engine.interner.display(&key[0]).to_string();
        let line = format!(
            "{} --{}--> {}",
            subj,
            m.engine.interner.display(&key[1]),
            m.engine.interner.display(&key[2])
        );
        let ov = topic_overlap(&tokens3(&line), &subj, &qt);
        // the embedding cache is keyed by render_fact format — the same
        // fact as Retrieval indexes it
        let render_line = format!(
            "current({}, {}, {})",
            subj,
            m.engine.interner.display(&key[1]),
            m.engine.interner.display(&key[2])
        );
        let sem = match (&qvec, embeds.get(&render_line)) {
            (Some(q), Some(v)) if !q.is_empty() && !v.is_empty() => {
                let mut c = lemmalog::semantics::cosine_pub(v, q) as f64;
                // the question asks about the user's own state; assistant
                // recommendations and third parties rank below first-person
                // facts at the same similarity
                if subj == "the_user" {
                    c += 0.15;
                }
                c
            }
            _ => 0.0,
        };
        let score = if ov >= 2 { 2.0 + ov as f64 } else { sem };
        if ov >= 2 || sem > 0.45 {
            scored.push((score, line));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<String> = Vec::new();
    for (_, l) in scored {
        if out.len() >= 12 {
            break;
        }
        if !out.contains(&l) {
            out.push(l);
        }
    }
    out
}

/// Count aggregates for counting-shaped questions: dynamic cnt_<rel>
/// rules over current(S, R, O), rendered with their member lists. An
/// additive section — the caller keeps every other context section.
fn count_section(m: &mut AgentMemory<MockExtractor>, question: &str) -> String {
    // dynamic count rules per distinct relation in current(S, R, O):
    // cnt_<rel>(S, count(O))
    let mut rels: Vec<String> = Vec::new();
    for key in m.engine.relation_keys("current") {
        if key.len() == 3 {
            let r = m.engine.interner.display(&key[1]).to_string();
            if !rels.contains(&r) {
                rels.push(r);
            }
        }
    }
    let mut rules = String::new();
    for r in &rels {
        let safe = r.replace('/', "_");
        rules.push_str(&format!(
            "cnt_{safe}(S, count(O)) :- current(S, \"{r}\", O).\n"
        ));
    }
    if !rules.is_empty() {
        let _ = m.engine.install_program(&rules);
        let now = m.engine.now;
        let _ = m.maintain(now);
    }
    // per-relation SUMs over numeric facts: "most money at which store"
    // is a grouped sum, which the aggregation engine computes exactly
    let mut sum_rules = String::new();
    let mut numeric_rels: Vec<String> = Vec::new();
    for key in m.engine.relation_keys("numeric") {
        if key.len() == 3 {
            let r = m.engine.interner.display(&key[1]).to_string();
            if !numeric_rels.contains(&r) {
                numeric_rels.push(r);
            }
        }
    }
    for r in &numeric_rels {
        // summing stated counts/totals is meaningless — they are already
        // aggregates; only true amounts (price, spent, ...) sum
        if r.contains("count") || r.contains("total") || r.contains("number") {
            continue;
        }
        let safe = r.replace('/', "_");
        sum_rules.push_str(&format!(
            "sum_{safe}(S, sum(N)) :- numeric(S, \"{r}\", N).\n"
        ));
    }
    if !sum_rules.is_empty() {
        let _ = m.engine.install_program(&sum_rules);
        let now = m.engine.now;
        let _ = m.maintain(now);
    }
    if std::env::var("LEMMALOG_DEBUG").is_ok() {
        let preds: Vec<String> = m
            .engine
            .relations
            .keys()
            .filter(|p| p.starts_with("cnt_"))
            .cloned()
            .collect();
        for p in &preds {
            eprintln!(
                "debug: {p} rows={} user_rows={}",
                m.engine.relation_keys(p).len(),
                m.engine
                    .relation_keys(p)
                    .iter()
                    .filter(|k| k.len() == 2 && m.engine.interner.display(&k[0]) == "the_user")
                    .count()
            );
        }
    }

    // assemble: counts section first (small), then facts, then generous
    // episode text from BM25 over the question
    let mut out = String::from("== counts (derived aggregates) ==\n");
    let mut count_lines = 0;
    let user_sym = m.engine.sym("the_user");
    let qtokens = tokens3(question);
    let preds: Vec<String> = m
        .engine
        .relations
        .keys()
        .filter(|p| p.starts_with("cnt_"))
        .cloned()
        .collect();
    for pred in &preds {
        for key in m.engine.relation_keys(pred) {
            if key.len() == 2 && key[0] == user_sym {
                // relevance filter: the count line (relation name) must
                // share a stemmed token with the question
                let line = m.engine.render_fact(pred, &key);
                let shares = tokens3(&line).iter().any(|t| t.len() >= 3 && qtokens.contains(t));
                if !shares {
                    continue;
                }
                // enumerate the counted members: the reader undercounts
                // when it only sees part of the set in the prose below
                let rel = pred.trim_start_matches("cnt_").to_string();
                let mut members: Vec<String> = Vec::new();
                for ck in m.engine.relation_keys("current") {
                    if ck.len() == 3
                        && ck[0] == user_sym
                        && m.engine.interner.display(&ck[1]) == rel
                    {
                        members.push(m.engine.interner.display(&ck[2]).to_string());
                    }
                }
                // merge name variants: "black Fender Stratocaster" is the
                // same instrument as "black Fender Stratocaster electric
                // guitar" — containment dedup keeps the fuller name
                let mut merged: Vec<String> = Vec::new();
                for mem in &members {
                    let lm = mem.to_lowercase();
                    let contained = merged
                        .iter()
                        .any(|k| k.to_lowercase().contains(&lm) || lm.contains(&k.to_lowercase()));
                    if !contained {
                        merged.push(mem.clone());
                    }
                }
                // consumed members: an item added to a set and later
                // watched/read/sold/removed is not current membership —
                // "to-watch list" counts what is still pending
                const CONSUMED: [&str; 10] = [
                    "watched", "read", "finished", "sold", "removed",
                    "gave", "donated", "completed", "returned", "cancel",
                ];
                let consumed: Vec<String> = merged
                    .iter()
                    .filter(|mem| {
                        let mt = tokens3(mem);
                        m.engine.relation_keys("current").iter().any(|k| {
                            if k.len() != 3 {
                                return false;
                            }
                            let rel = m.engine.interner.display(&k[1]).to_lowercase();
                            if !CONSUMED.iter().any(|c| rel.contains(c)) {
                                return false;
                            }
                            let obj_toks = tokens3(&m.engine.interner.display(&k[2]));
                            mt.iter().any(|t| obj_toks.contains(t))
                        })
                    })
                    .cloned()
                    .collect();
                let n_raw = key[1].as_int().unwrap_or(0);
                let current_est = merged.len().saturating_sub(consumed.len());
                if !consumed.is_empty() && current_est > 0 {
                    out.push_str(&format!(
                        "{line} — {} extracted, {} already consumed (watched/read/sold/…), current membership ≈ {}: {}\n",
                        n_raw,
                        consumed.len(),
                        current_est,
                        merged
                            .iter()
                            .filter(|x| !consumed.contains(x))
                            .take(10)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                } else if merged.len() > 1 && (merged.len() as i64) < n_raw {
                    out.push_str(&format!(
                        "{line} — extracted as {} rows, {} distinct after merging name variants: {}\n",
                        n_raw,
                        merged.len(),
                        merged
                            .iter()
                            .take(10)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                } else if !merged.is_empty() {
                    out.push_str(&format!(
                        "{line}: {}\n",
                        merged.iter().take(10).cloned().collect::<Vec<_>>().join("; ")
                    ));
                } else {
                    out.push_str(&format!("{line}\n"));
                }
                count_lines += 1;
            }
        }
    }
    // per-relation SUMs over normalized numeric facts: "most money at
    // which store" is a grouped sum, which the aggregation engine
    // computes exactly
    let mut sum_lines = 0;
    let sum_preds: Vec<String> = m
        .engine
        .relations
        .keys()
        .filter(|p| p.starts_with("sum_"))
        .cloned()
        .collect();
    for pred in &sum_preds {
        for key in m.engine.relation_keys(pred) {
            if key.len() == 2 && key[0] == user_sym {
                let line = m.engine.render_fact(pred, &key);
                let shares = tokens3(&line).iter().any(|t| t.len() >= 3 && qtokens.contains(t));
                if !shares {
                    continue;
                }
                let rel = pred.trim_start_matches("sum_").to_string();
                let members: Vec<String> = m
                    .engine
                    .relation_keys("numeric")
                    .into_iter()
                    .filter(|k| {
                        k.len() == 3
                            && k[0] == user_sym
                            && m.engine.interner.display(&k[1]) == rel
                    })
                    .map(|k| m.engine.interner.display(&k[2]).to_string())
                    .collect();
                out.push_str(&format!(
                    "{line} ({} entries: {})\n",
                    members.len(),
                    members.join(", ")
                ));
                sum_lines += 1;
            }
        }
    }
    count_lines += sum_lines;
    // stated counts: the transcript itself often says "my list has 20
    // items" — extracted as a numeric fact, it is the most direct
    // evidence for how-many questions; latest assertion wins
    {
        let mut stated: Vec<(i64, String)> = Vec::new();
        for key in m.engine.relation_keys("edge") {
            if key.len() != 6 {
                continue;
            }
            let rel = m.engine.interner.display(&key[1]).to_lowercase();
            if !(rel.contains("count") || rel.contains("total") || rel.contains("number")) {
                continue;
            }
            let obj = m.engine.interner.display(&key[2]).to_string();
            if !obj.chars().all(|c| c.is_ascii_digit()) || obj.is_empty() {
                continue;
            }
            let ts = key[5].as_int().unwrap_or(0);
            let rel_disp = m.engine.interner.display(&key[1]).to_string();
            let shares = tokens3(&rel_disp).iter().any(|t| t.len() >= 3 && qtokens.contains(t));
            if !shares {
                continue;
            }
            stated.push((ts, format!("stated {} = {}", rel_disp, obj)));
        }
        stated.sort_by(|a, b| b.0.cmp(&a.0));
        if let Some((_, latest)) = stated.first() {
            out.push_str(&format!("{latest} (latest stated value)\n"));
            count_lines += 1;
        }
    }
    if count_lines == 0 {
        out.clear(); // no counts: the section is omitted entirely
    }
    out
}

/// Fact and question embeddings for the semantic rerank, with a
/// snapshot-side file cache ({snap}.embed.json) so the ~2K facts embed
/// once per conversation, not once per question. Returns None when the
/// embedding endpoint is off or unreachable (plain BM25 selection).
fn semantic_embeds(snap: &str, r: &Retrieval, question: &str) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
    let base = embed_base()?;
    let embedder = lemmalog::llm::HttpEmbedder::new(
        &base,
        "text-embedding-nomic-embed-text-v1.5",
    );
    let cache_path = format!("{snap}.embed.json");
    let mut cache: std::collections::HashMap<String, Vec<f32>> = std::fs::read_to_string(&cache_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let renders = r.fact_renders();
    let mut embeds: Vec<Vec<f32>> = Vec::with_capacity(renders.len());
    let mut missing = 0usize;
    for line in &renders {
        let v = match cache.get(line) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => {
                missing += 1;
                let v = embedder.embed(line);
                if v.is_empty() {
                    return None; // endpoint down: degrade, don't poison
                }
                cache.insert(line.clone(), v.clone());
                v
            }
        };
        embeds.push(v);
    }
    if missing > 0 {
        let _ = std::fs::write(&cache_path, serde_json::to_string(&cache).unwrap_or_default());
    }
    let q = embedder.embed(question);
    if q.is_empty() {
        return None;
    }
    Some((embeds, q))
}

/// Days since 1970-01-01 from a YYYYMMDD int (Howard Hinnant's
/// days_from_civil): lets date differences be real subtraction.
fn ymd_to_days(v: i64) -> i64 {
    let (y, m, d) = (v / 10000, (v / 100) % 100, v % 100);
    let (y, m) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn context(snap: &str, question: &str) -> String {
    let mut m = AgentMemory::load(MockExtractor::new(0.9), snap).expect("load snapshot");
    let _ = m.maintain(m.engine.now);
    let counting = is_counting_question(question);
    let counts = if counting {
        count_section(&mut m, question)
    } else {
        String::new()
    };
    let budget: usize = std::env::var("LEMMALOG_CONTEXT_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if counting { 3800 } else { 3200 });
    let r = Retrieval::build(&m.engine, m.episodes());
    let sel = match semantic_embeds(snap, &r, question) {
        Some((fe, q)) => r.select_semantic(question, budget, &fe, &q),
        None => r.select(question, budget),
    };
    let mut base = r.render(&sel);

    let dates: BTreeMap<String, String> = std::fs::read_to_string(format!("{snap}.dates.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    // reference date: relative time in the question ("two weeks ago")
    // is unanchored without it, and the reader needs it to do date
    // arithmetic against session dates
    let reference_date = dates
        .iter()
        .max_by_key(|(k, _)| k.parse::<i64>().unwrap_or(0))
        .map(|(_, v)| v.clone());
    if let Some(rd) = &reference_date {
        let head = format!("CURRENT DATE (most recent session): {rd}\n");
        base = format!("{head}{base}");
    }
    if !counts.is_empty() {
        // insert after the reference-date line, before the memory section
        if let Some(pos) = base.find('\n') {
            let (head, rest) = base.split_at(pos + 1);
            base = format!("{head}{counts}\n{rest}");
        } else {
            base = format!("{counts}\n{base}");
        }
    }
    let date_of = |v: &Value| -> String {
        match v.as_int() {
            Some(ts) if ts > 0 && ts != i64::MAX => dates
                .get(&ts.to_string())
                .cloned()
                .unwrap_or_else(|| ts.to_string()),
            _ => "open".to_string(),
        }
    };
    // budgeted dated-history append, whole-token subject matching
    let qtokens: std::collections::BTreeSet<String> =
        tokenize_pub(question).into_iter().filter(|t| t.len() >= 3).collect();
    let mut hist = String::new();
    for key in m.engine.relation_keys("edge") {
        if hist.len() > 700 * 4 {
            break;
        }
        let subj = m.engine.interner.display(&key[0]).to_string();
        let matches = tokenize_pub(&subj)
            .into_iter()
            .any(|t| t.len() >= 3 && qtokens.contains(&t));
        if !matches {
            continue;
        }
        let disp = |i: usize| m.engine.interner.display(&key[i]).to_string();
        hist.push_str(&format!(
            "{} --{}--> {}, {} .. {} (asserted {})\n",
            disp(0),
            disp(1),
            disp(2),
            date_of(&key[3]),
            date_of(&key[4]),
            date_of(&key[5]),
        ));
    }
    if !hist.is_empty() {
        base.push_str(&format!(
            "\nEDGE HISTORY: subject --relation--> object, valid_from .. valid_to (asserted at):\n{hist}"
        ));
    }
    // reconciled aliases: shown to the reader only at high confidence —
    // wrong equalities actively mislead, so the risky merges stay
    // retrieval-side (bridge boosts) and never render
    let alias_lines: Vec<String> = m
        .engine
        .relation_keys("alias")
        .iter()
        .filter_map(|k| {
            let conf = m
                .engine
                .fact("alias", k)
                .map(|f| f.ann.conf)
                .unwrap_or(0.0);
            if conf < 0.9 {
                return None;
            }
            let (a, b) = (
                m.engine.interner.display(&k[0]).to_string(),
                m.engine.interner.display(&k[1]).to_string(),
            );
            let relevant = tokenize_pub(&a)
                .iter()
                .chain(tokenize_pub(&b).iter())
                .any(|t| t.len() >= 3 && qtokens.contains(t));
            relevant.then(|| format!("{a} = {b}"))
        })
        .take(12)
        .collect();
    if !alias_lines.is_empty() {
        base.push_str(&format!(
            "\nENTITY ALIASES (same entity, reconciled):\n{}\n",
            alias_lines.join("\n")
        ));
    }
    // mention timeline: when question entities FIRST appeared in the
    // conversation. Explicitly labeled as mention order — mention time is
    // not event time (a later turn can describe an earlier event)
    let mut first_seen: BTreeMap<String, i64> = BTreeMap::new();
    for key in m.engine.relation_keys("edge") {
        let ts = key[5].as_int().unwrap_or(i64::MAX);
        for pos in [0usize, 2] {
            if let Value::Sym(_) = &key[pos] {
                let n = m.engine.interner.display(&key[pos]).to_string();
                if !n.chars().any(|c| c.is_alphabetic()) {
                    continue;
                }
                let e = first_seen.entry(n).or_insert(ts);
                if ts < *e {
                    *e = ts;
                }
            }
        }
    }
    let mut timeline: Vec<(i64, String)> = first_seen
        .iter()
        .filter(|(n, _)| {
            tokenize_pub(n)
                .into_iter()
                .any(|t| t.len() >= 3 && qtokens.contains(&t))
        })
        .map(|(n, ts)| (*ts, n.clone()))
        .collect();
    timeline.sort();
    if !timeline.is_empty() {
        let lines: Vec<String> = timeline
            .iter()
            .take(15)
            .map(|(ts, n)| {
                let d = dates
                    .get(&ts.to_string())
                    .cloned()
                    .unwrap_or_else(|| ts.to_string());
                format!("{n} first mentioned {d}")
            })
            .collect();
        base.push_str(&format!(
            "\nMENTION TIMELINE (first mention in conversation; mention order is NOT event order):\n{}\n",
            lines.join("\n")
        ));
    }
    // attribution contrast: which subjects hold facts on this question's
    // topic, and which question-mentioned parties hold none — the zero
    // count is the misattribution signal ("gift to Melanie" when the
    // story is Caroline's shows Melanie: 0)
    {
        let qt = tokens3(question);
        let mut holders: BTreeMap<String, usize> = BTreeMap::new();
        let mut subjects: Vec<String> = Vec::new();
        for key in m.engine.relation_keys("current") {
            if key.len() != 3 {
                continue;
            }
            let subj = m.engine.interner.display(&key[0]).to_string();
            if !subjects.contains(&subj) {
                subjects.push(subj.clone());
            }
            let line = format!(
                "{} --{}--> {}",
                subj,
                m.engine.interner.display(&key[1]),
                m.engine.interner.display(&key[2])
            );
            if topic_overlap(&tokens3(&line), &subj, &qt) >= 1 {
                *holders.entry(subj).or_insert(0) += 1;
            }
        }
        // question-mentioned parties (by subject-name token match)
        let asked: Vec<String> = subjects
            .iter()
            .filter(|s| {
                let st = tokens3(s);
                st.iter().any(|t| t.len() >= 3 && qt.contains(t))
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
    // date facts with precomputed differences: the reader reliably states
    // both dates and then fails to subtract — the engine owns the
    // arithmetic
    let q_lower = question.to_lowercase();
    let wants_dates = [
        "how many days", "how many months", "how long", "before", "after",
        "ago", "between", "when did", "what date", "which day",
    ]
    .iter()
    .any(|p| q_lower.contains(p));
    if wants_dates {
        let qt = tokens3(question);
        let tok_match = |a: &str, b: &str| -> bool {
            a == b || (a.len() >= 4 && b.len() >= 4 && (a.starts_with(b) || b.starts_with(a)))
        };
        // date-bearing current facts matched to the question by tokens
        let mut date_facts: Vec<(String, i64)> = Vec::new();
        for key in m.engine.relation_keys("current") {
            if key.len() != 3 {
                continue;
            }
            let obj = m.engine.interner.display(&key[2]).to_string();
            let Some(ymd) = lemmalog::longmemeval::date_to_int(&obj) else {
                continue;
            };
            let line = format!(
                "{} --{}--> {}",
                m.engine.interner.display(&key[0]),
                m.engine.interner.display(&key[1]),
                obj
            );
            let matched = tokens3(&line)
                .iter()
                .any(|t| t.len() >= 3 && qt.iter().any(|q| tok_match(q, t)));
            if matched && !date_facts.iter().any(|(l, _)| l == &line) {
                date_facts.push((line, ymd));
            }
        }
        date_facts.truncate(6);
        if !date_facts.is_empty() {
            let mut sec = String::from(
                "\nDATE FACTS (facts carrying dates; differences computed):\n",
            );
            // reference ymd from the sidecar date string ("2023/05/30 ...")
            let ref_ymd = reference_date.as_ref().and_then(|d| {
                d.split_whitespace()
                    .next()
                    .and_then(|p| {
                        let dp: Vec<i64> =
                            p.split('/').filter_map(|x| x.parse().ok()).collect();
                        (dp.len() == 3)
                            .then(|| dp[0] * 10000 + dp[1] * 100 + dp[2])
                    })
            });
            for (line, ymd) in &date_facts {
                if let Some(r) = ref_ymd {
                    let dd = ymd_to_days(r) - ymd_to_days(*ymd);
                    if dd > 0 {
                        sec.push_str(&format!(
                            "{line} — {} days before current date (~{} months ago)\n",
                            dd,
                            (dd as f64 / 30.44).round()
                        ));
                        continue;
                    }
                }
                sec.push_str(&format!("{line}\n"));
            }
            let mut pairs = 0;
            for i in 0..date_facts.len() {
                for j in i + 1..date_facts.len() {
                    if pairs >= 6 {
                        break;
                    }
                    let (a, b) = (&date_facts[i], &date_facts[j]);
                    let dd = (ymd_to_days(b.1) - ymd_to_days(a.1)).abs();
                    sec.push_str(&format!(
                        "{} vs {}: {} days apart (~{} months)\n",
                        a.0,
                        b.0,
                        dd,
                        (dd as f64 / 30.44).round()
                    ));
                    pairs += 1;
                }
            }
            base.push_str(&sec);
        }
    }
    // current-state latest values: when a slot (subject + relation) holds
    // several open values, the newest by valid_from is the current one —
    // render the supersession explicitly so "what is my current X"
    // questions don't drown in history
    if q_lower.contains("current") || q_lower.contains(" now") || q_lower.contains("latest") {
        let qt_cur = tokens3(question);
        let mut slots: BTreeMap<(String, String), Vec<(i64, String)>> = BTreeMap::new();
        for key in m.engine.relation_keys("edge") {
            if key.len() != 6 {
                continue;
            }
            let subj = m.engine.interner.display(&key[0]).to_string();
            let rel = m.engine.interner.display(&key[1]).to_string();
            let line = format!("{subj} --{rel}--> {}", m.engine.interner.display(&key[2]));
            // only slots the question touches
            if !tokens3(&line).iter().any(|t| t.len() >= 3 && qt_cur.contains(t)) {
                continue;
            }
            let vf = key[3].as_int().unwrap_or(0);
            if vf == i64::MAX {
                continue;
            }
            slots
                .entry((subj, rel))
                .or_default()
                .push((vf, m.engine.interner.display(&key[2]).to_string()));
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
            let newest = &distinct[0];
            let older: Vec<String> = distinct.iter().skip(1).take(3).cloned().collect();
            sec.push_str(&format!(
                "  {subj} --{rel}--> {newest} (current; superseded: {})\n",
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
/// Recall fallback: when the memory lacks the answer, run ONE targeted
/// extraction pass over the BM25-top episodes WITH the question in the
/// prompt — the question tells the extractor exactly what to hunt for.
/// Returns the extracted triples as context lines (empty if nothing found).
fn recall(snap: &str, question: &str) -> String {
    let m = AgentMemory::load(MockExtractor::new(0.9), snap).expect("load snapshot");
    // rank episodes by BM25 over the question, take the top 5
    let r = lemmalog::retrieval::Retrieval::build(&m.engine, m.episodes());
    let sel = r.select(question, 1200);
    let mut episodes_text = String::new();
    for &i in &sel.episodes {
        let ep = &m.episodes()[i];
        episodes_text.push_str(&format!("[{}] {}\n", ep.id, ep.text));
    }
    if episodes_text.is_empty() {
        return String::new();
    }
    // one targeted extraction call (Claude), strict-parsed
    let base = client();
    let prompt = format!(
        "The user asks: {question}\n\nBelow are the most relevant source \
episodes. Extract ONLY the facts that answer this question, in the form \
S --rel--> O (confidence omitted). If the episodes do not contain the \
answer, reply with the single word NONE.\n\n{episodes_text}\n\nTriples only:"
    );
    let Ok(response) = base.chat(
        "You extract factual triples precisely. Output only triple lines \
or NONE.",
        &prompt,
    ) else {
        return String::new();
    };
    if response.trim().eq_ignore_ascii_case("none") {
        return String::new();
    }
    let facts = lemmalog::agent::parse_protocol_strict(&response, 0.8);
    facts
        .iter()
        .map(|f| format!("{} --{}--> {}\n", f.subj, f.pred, f.obj))
        .collect()
}
