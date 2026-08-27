//! lemmalog-bench: the bridge binary for the MemEval harness.
//!
//! Two commands:
//!   lemmalog-bench ingest  <conv.json> <snapshot-path>   # extraction (Claude,
//!     chunked, file-cached) + update policy + derived rules -> snapshot
//!   lemmalog-bench context <snapshot-path> <question>    # hybrid retrieval ->
//!     prints the assembled memory context for the standardized reader
//!
//! The conv.json format is MemEval's normalized LoCoMo shape (session_N keys
//! with {speaker, text} turns and session_N_date_time).
//!
//! Env: ANTHROPIC_API_KEY (extraction), LEMMALOG_EXTRACT_MODEL (default
//! claude-sonnet-4-6), LEMMALOG_CACHE_DIR (extraction cache: pay once,
//! rerun free), LEMMALOG_CONTEXT_BUDGET (default 1800).

#![cfg(feature = "llm")]

use lemmalog::agent::{AgentMemory, Episode, MockExtractor};
use lemmalog::llm::{LlmClientExtractor, OpenAiClient};
use lemmalog::retrieval::tokenize_pub;
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
        _ => {
            eprintln!("usage: lemmalog-bench ingest|context ...");
            std::process::exit(2);
        }
    }
}

/// "2023/04/10 (Mon) 17:50" -> monotonic minutes (matches the runner).
fn parse_date(s: &str) -> i64 {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let (d, t) = (
        parts.first().copied().unwrap_or("0/0/0"),
        parts.get(3).copied().unwrap_or("0:0"),
    );
    let dp: Vec<i64> = d.split('/').filter_map(|x| x.parse().ok()).collect();
    let tp: Vec<i64> = t.split(':').filter_map(|x| x.parse().ok()).collect();
    (((dp.first().copied().unwrap_or(0) * 366
        + dp.get(1).copied().unwrap_or(0) * 31
        + dp.get(2).copied().unwrap_or(0))
        * 24
        + tp.first().copied().unwrap_or(0))
        * 60)
        + tp.get(1).copied().unwrap_or(0)
}

const RULES: &str = "\
current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.\n\
reports_to(X,Y) :- current(X,\"manager\",Y).\n\
trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).\n\
happened_before(A, B) :- dated(A, D1), dated(B, D2), D1 < D2.\n\
stated_before(X, Y) :- current(X, \"before\", Y).\n\
stated_before(X, Z) :- stated_before(X, Y), stated_before(Y, Z).\n";

fn client() -> OpenAiClient {
    let model = std::env::var("LEMMALOG_EXTRACT_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-6".to_string());
    OpenAiClient::new("https://api.anthropic.com/v1", &model, "")
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
    // date-shaped relations -> dated/2 (the runner's data-driven ordering)
    let mut order_rules = String::new();
    let mut checked: Vec<String> = Vec::new();
    for key in m.engine.relation_keys("current") {
        if key.len() != 3 {
            continue;
        }
        let r = m.engine.interner.display(&key[1]).to_string();
        if checked.contains(&r) {
            continue;
        }
        checked.push(r.clone());
        let obj = m.engine.interner.display(&key[2]);
        if obj.len() == 10 && obj.as_bytes()[4] == b'-' && obj.as_bytes()[7] == b'-' {
            order_rules.push_str(&format!("dated(S, D) :- current(S, \"{r}\", D).\n"));
        }
    }
    if !order_rules.is_empty() {
        let _ = m.engine.install_program(&order_rules);
    }
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

fn context(snap: &str, question: &str) -> String {
    let mut m = AgentMemory::load(MockExtractor::new(0.9), snap).expect("load snapshot");
    let _ = m.maintain(m.engine.now);
    let budget: usize = std::env::var("LEMMALOG_CONTEXT_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800);
    let base = m.context_for_query(question, budget);

    let dates: BTreeMap<String, String> = std::fs::read_to_string(format!("{snap}.dates.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let date_of = |v: &lemmalog::Value| -> String {
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
    if hist.is_empty() {
        base
    } else {
        format!(
            "{base}\nEDGE HISTORY: subject --relation--> object, valid_from .. valid_to (asserted at):\n{hist}"
        )
    }
}
