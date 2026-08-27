//! The LLM as rule author: seed a memory with facts (no model calls), ask
//! the live model to write a recursive rule for a natural-language request,
//! validate + install + backfill, then verify against ground truth with
//! pure engine queries.
//!
//!     cargo run --features llm --example llm_rules [model] [base_url]

use lemmalog::agent::{AgentMemory, MockExtractor};
use lemmalog::llm::{parse_rule_candidates, OpenAiClient};
use std::time::Instant;

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

    let mut m = AgentMemory::new(MockExtractor::new(0.9), "").unwrap();
    // deterministic seed facts (the `--rel-->` protocol, no model calls)
    let seed = "\
alice --works_at--> acme
bob --works_at--> acme
carol --works_at--> gigant
dana --works_at--> gigant
alice --manager--> bob
bob --manager--> carol";
    m.observe_at(seed, 100);
    m.maintain(100);
    println!("seeded memory:\n{}", m.engine.schema_summary());

    let request = "People belong to an organization if they work there, or \
if they report (directly or transitively) to someone who works there. \
Write rule(s) defining org_member(Person, Org).";
    println!("\n== asking {model} to author rules ==\nrequest: {request}");

    let client = OpenAiClient::new(&base, &model, "");
    let schema = m.engine.schema_summary();
    let raw = client
        .synthesize_rules(&schema, "(none yet)", request)
        .or_else(|e1| {
            eprintln!("first attempt failed ({e1}); retrying...");
            client.synthesize_rules(&schema, "(none yet)", request)
        })
        .unwrap_or_else(|e| {
            eprintln!("synthesis failed: {e}");
            std::process::exit(1);
        });
    println!("\nmodel output:\n{raw}");

    let lines = match parse_rule_candidates(&raw) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("validation rejected: {e}");
            std::process::exit(1);
        }
    };
    let program = lines.join("\n");
    println!("\nvalidated rules:\n{program}");

    let batch = m
        .install_rules(&program)
        .unwrap_or_else(|e| {
            eprintln!("install rejected: {e}");
            std::process::exit(1);
        });
    println!("installed batch {batch}");
    let derived = m.maintain(100);
    println!("backfill derived {derived} facts");

    // ground truth: alice/bob under bob->carol chain...
    // alice works acme; alice->bob (acme); bob->carol (gigant) =>
    // alice: {acme, gigant}, bob: {acme, gigant}, carol: {gigant}, dana: {gigant}
    let expect: Vec<(&str, Vec<&str>)> = vec![
        ("alice", vec!["acme", "gigant"]),
        ("bob", vec!["acme", "gigant"]),
        ("carol", vec!["gigant"]),
        ("dana", vec!["gigant"]),
    ];
    let mut pass = true;
    println!("\n== verification (pure engine, no model) ==");
    for (person, orgs) in &expect {
        let p = m.engine.sym(person);
        let mut got: Vec<String> = m
            .engine
            .query("org_member", &[Some(p), None])
            .into_iter()
            .map(|(k, _)| m.engine.interner.display(&k[1]).to_string())
            .collect();
        got.sort();
        let mut want: Vec<&str> = orgs.clone();
        want.sort();
        let ok = got == want;
        pass &= ok;
        println!(
            "org_member({person}) = {got:?} want {want:?} [{}]",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // revertability
    m.uninstall_rules(&batch);
    m.maintain(100);
    let left = m.engine.query("org_member", &[]).len();
    println!(
        "\nuninstalled {batch}: org_member rows = {left} (reverted: {})",
        left == 0
    );

    println!(
        "\n== {} | wall {:.0}s ==",
        if pass { "RULE AUTHORING PASS" } else { "RULE AUTHORING FAIL" },
        t0.elapsed().as_secs_f64()
    );
}
