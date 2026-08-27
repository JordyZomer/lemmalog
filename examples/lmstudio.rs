//! Real-model end-to-end: natural-language sessions -> live extraction ->
//! datalog memory -> grounded Q&A, scored. Also registers entities with
//! real embeddings (nomic) for hybrid retrieval.
//!
//!     cargo run --features llm --example lmstudio [model] [base_url]
//!
//! Defaults: model qwen3.8-27b-uncensored-mlx, base http://localhost:1234/v1,
//! embedder text-embedding-nomic-embed-text-v1.5. The model thinks for
//! minutes per call; the whole run takes ~30 minutes.

use lemmalog::agent::AgentMemory;
use lemmalog::llm::{LlmClientExtractor, OpenAiClient};
use lemmalog::llm::HttpEmbedder;
use lemmalog::semantics::SemanticIndex;
use lemmalog::Value;
use std::time::Instant;

const RULES: &str = "\
reports_to(X,Y) :- current(X,\"manager\",Y).\n\
trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).\n";

const SESSIONS: [(i64, &str); 4] = [
    (100, "Hey! I am Alice. I just started at Acme Corp last week, doing cloud infrastructure. My manager is Bob, and Bob reports to Carol, the VP of engineering."),
    (200, "Quick update - I left Acme Corp. I am now at Gigant Systems doing platform engineering. Also I really love hiking and Zeta Analytics products."),
    (300, "Still true that I like Acme Corp's products too, by the way. Both brands are great."),
    (400, "Dana is my manager at Gigant Systems. Carol has been really helpful mentoring me."),
];

const ENTITIES: [(&str, &str); 7] = [
    ("Alice", "Alice, employee, cloud and platform engineer"),
    ("Bob", "Bob, engineering manager at Acme Corp"),
    ("Carol", "Carol, VP of engineering, mentor"),
    ("Dana", "Dana, manager at Gigant Systems"),
    ("Acme Corp", "Acme Corp, company, cloud products"),
    ("Gigant Systems", "Gigant Systems, company, platform engineering"),
    ("Zeta Analytics", "Zeta Analytics, analytics products company"),
];

struct Question {
    q: &'static str,
    must_contain: &'static [&'static str],
}

const QUESTIONS: [Question; 4] = [
    Question {
        q: "Which company does Alice currently work at?",
        must_contain: &["gigant"],
    },
    Question {
        q: "Who does Alice report to, directly or through her manager chain?",
        must_contain: &["bob", "carol", "dana"],
    },
    Question {
        q: "Which companies' products does Alice like?",
        must_contain: &["acme", "zeta"],
    },
    Question {
        q: "Where did Alice work before Gigant Systems?",
        must_contain: &["acme"],
    },
];

fn main() {
    let t0 = Instant::now();
    let model = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("LEMMALOG_MODEL").ok())
        .unwrap_or_else(|| "qwen3.8-27b-uncensored-mlx".to_string());
    let base = std::env::args()
        .nth(2)
        .or_else(|| std::env::var("LEMMALOG_URL").ok())
        .unwrap_or_else(|| "http://localhost:1234/v1".to_string());
    let embed_model = "text-embedding-nomic-embed-text-v1.5";
    // embeddings stay local (nomic on LM Studio); chat goes to the model url
    let embed_base = std::env::var("LEMMALOG_EMBED_URL")
        .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());

    println!("== lemmalog live: model={model} base={base} embed={embed_base} ==");
    let client = OpenAiClient::new(&base, &model, embed_model);
    let mut m = AgentMemory::new(LlmClientExtractor::new(OpenAiClient::new(&base, &model, embed_model)), RULES)
        .unwrap();

    // ---- ingestion: live extraction per session ----
    for (ts, text) in SESSIONS {
        let t = Instant::now();
        let epoch_before = m.engine.epoch();
        let report = m.observe_as(text, ts, "Alice");
        let derived = m.maintain(ts);
        println!(
            "\n[session t={ts}] ({:.0}s) added={} updated={} noop={} derived={}",
            t.elapsed().as_secs_f64(),
            report.added,
            report.updated,
            report.noop,
            derived
        );
        // dump every fact this session added or changed (change feed)
        let feed = m.engine.changes_since(epoch_before);
        let mut shown = 0;
        for ch in &feed {
            if let lemmalog::Change::Added(_, (pred, key)) = ch {
                if pred == "edge" || pred == "current" {
                    println!("  new: {}", m.engine.render_fact(pred, key));
                    shown += 1;
                }
            }
        }
        let _ = shown;
        for e in &report.escalations {
            println!("  escalation: {e}");
        }
    }

    // ---- semantic index with real embeddings ----
    println!("\n== embedding entity registry (nomic) ==");
    let embedder = HttpEmbedder::new(&embed_base, embed_model);
    let mut index = SemanticIndex::new(embedder);
    for (name, profile) in ENTITIES {
        index.register(name, profile);
    }
    println!("registered {} entities", ENTITIES.len());

    // ---- grounded Q&A ----
    println!("\n== questions (grounded answers, live model) ==");
    let mut score = 0usize;
    for Question { q, must_contain } in QUESTIONS.iter() {
        let t = Instant::now();
        // hybrid retrieval: entities linked to the question by embedding
        let linked = index.search(q, 4);
        let names: Vec<&str> = linked.iter().map(|(n, _)| n.as_str()).collect();
        let mut ctx = String::new();
        ctx.push_str("MEMORY:\n");
        for name in &names {
            let v = m.engine.sym(name);
            for (k, ann) in m.engine.query("current", &[Some(v), None, None]) {
                ctx.push_str(&format!(
                    "current: {} (conf {:.2})\n",
                    m.engine.render_fact("current", &k),
                    ann.conf
                ));
            }
            // full employment history incl. superseded intervals
            for (k, _) in m.engine.query("edge", &[Some(v), None, None, None, None, None]) {
                if matches!(k[1], Value::Sym(ref r) if m.engine.interner.resolve(*r) == "works_at") {
                    ctx.push_str(&format!("history: {}\n", m.engine.render_fact("edge", &k)));
                }
            }
        }
        let alice_sym = m.engine.sym("Alice");
        for row in m.engine.query("reports_to", &[Some(alice_sym), None]) {
            ctx.push_str(&format!("derived: {}\n", m.engine.render_fact("reports_to", &row.0)));
        }
        ctx.push_str("\n(An edge's 5th field is valid-to: 9223372036854775807 means still true.)");

        let answer = client
            .chat(
                "Answer ONLY from the MEMORY block. If the memory does not contain the answer, say 'unknown'. Be concise: one sentence.",
                &format!("MEMORY:\n{ctx}\n\nQUESTION: {q}"),
            )
            .unwrap_or_else(|e| format!("(answer failed: {e})"));
        let norm = answer.to_lowercase();
        let pass = must_contain.iter().all(|needle| norm.contains(needle));
        if pass {
            score += 1;
        }
        println!(
            "\nQ: {q}\n  linked entities: {names:?}\n  A ({:.0}s): {answer}\n  [{}] expected substrings {must_contain:?}",
            t.elapsed().as_secs_f64(),
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // ---- deterministic memory artifacts (no model calls) ----
    println!("\n== proof tree (pure engine, no model) ==");
    let (a, wa, gig) = (
        m.engine.sym("Alice"),
        m.engine.sym("works_at"),
        m.engine.sym("Gigant Systems"),
    );
    let hist = m.engine.query("edge", &[Some(a), Some(wa), Some(gig), None, None, None]);
    if let Some((k, _)) = hist.first() {
        print!("{}", m.engine.why("edge", k));
    }
    let (rows, added) = m
        .what_if("Eve --manager--> Alice", "reports_to(\"Eve\", Y)")
        .unwrap();
    println!("\nwhat_if Eve managed Alice => {rows:?} (would add {added} facts; store untouched)");

    let extractor = m.extractor_stats();
    println!(
        "\n== score: {}/{} == extractor calls: {} failures: {} | wall: {:.0}s ==",
        score,
        QUESTIONS.len(),
        extractor.0,
        extractor.1,
        t0.elapsed().as_secs_f64()
    );
}
