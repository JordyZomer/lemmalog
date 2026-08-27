//! LongMemEval oracle split through the lemmalog memory pipeline vs the
//! transcript-only baseline, scored with SQuAD-style token F1 per question
//! type.
//!
//!     cargo run --release --features llm --example longmemeval \
//!         [n_per_type] [model] [base_url]
//!
//! Expects data/longmemeval_oracle.json and ANTHROPIC_API_KEY (or
//! LEMMALOG_API_KEY) when pointed at a hosted endpoint.

use lemmalog::llm::OpenAiClient;
use lemmalog::longmemeval::{load, run_instance};
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();
    let n_per_type: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let model = std::env::args()
        .nth(2)
        .or_else(|| std::env::var("LEMMALOG_MODEL").ok())
        .unwrap_or_else(|| "claude-opus-4-8".to_string());
    let base = std::env::args()
        .nth(3)
        .or_else(|| std::env::var("LEMMALOG_URL").ok())
        .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string());
    let data = std::env::var("LEMMALOG_DATA")
        .unwrap_or_else(|_| "data/longmemeval_oracle.json".to_string());

    let instances = load(&data).expect("load dataset");
    println!(
        "== LongMemEval (oracle): {} instances, {} per type, model {model} ==",
        instances.len(),
        n_per_type
    );

    // selection: LEMMALOG_Q=substr,substr filters by question text
    // (comma-separated, any match); otherwise first n per type
    let mut selection = Vec::new();
    if let Ok(filter) = std::env::var("LEMMALOG_Q") {
        let needles: Vec<String> = filter
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        for inst in &instances {
            if needles.iter().any(|n| inst.question.to_lowercase().contains(n)) {
                selection.push(inst);
            }
        }
        println!("LEMMALOG_Q filter: {} instances", selection.len());
    } else {
        let mut picked: BTreeMap<&str, usize> = BTreeMap::new();
        for inst in &instances {
            let c = picked.entry(inst.question_type.as_str()).or_insert(0);
            if *c < n_per_type {
                *c += 1;
                selection.push(inst);
            }
        }
    }

    let chat = OpenAiClient::new(&base, &model, "");
    let mut per_type: BTreeMap<String, (usize, f64, f64, usize, usize, usize, usize)> =
        BTreeMap::new();
    for (i, inst) in selection.iter().enumerate() {
        let extract_client = OpenAiClient::new(&base, &model, "");
        let t = Instant::now();
        match run_instance(extract_client, &chat, inst) {
            Ok(r) => {
                println!(
                    "\n[{}/{}] {} ({}s)",
                    i + 1,
                    selection.len(),
                    inst.question_type,
                    t.elapsed().as_secs_f64()
                );
                println!("  Q: {}", inst.question);
                println!("  gold:      {}", r.gold);
                println!("  memory:    {} (F1 {:.2})", r.memory_pred, r.memory_f1);
                println!("  baseline:  {} (F1 {:.2})", r.baseline_pred, r.baseline_f1);
                println!(
                    "  tokens: memory {} vs transcript {} ({:.1}x)",
                    r.memory_ctx_tokens,
                    r.transcript_tokens,
                    if r.memory_ctx_tokens > 0 {
                        r.transcript_tokens as f64 / r.memory_ctx_tokens as f64
                    } else {
                        0.0
                    }
                );
                let e = per_type.entry(r.question_type.clone()).or_insert((0, 0.0, 0.0, 0, 0, 0, 0));
                e.0 += 1;
                e.1 += r.memory_f1;
                e.2 += r.baseline_f1;
                e.3 += r.memory_em as usize;
                e.4 += r.baseline_em as usize;
                e.5 += r.memory_ctx_tokens;
                e.6 += r.transcript_tokens;
            }
            Err(e) if e == "__no_answer__" => {
                println!("\n[{}/{}] {} (context assembled, answers skipped)", i + 1, selection.len(), inst.question_type);
            }
            Err(e) => println!("\n[{}/{}] {} FAILED: {e}", i + 1, selection.len(), inst.question_type),
        }
    }

    println!("\n=== summary (n, memory F1, baseline F1, memory EM, baseline EM) ===");
    let (mut n_all, mut mf, mut bf, mut me, mut be) = (0usize, 0.0, 0.0, 0, 0);
    for (t, (n, m, b, em, eb, _tk, _tt)) in &per_type {
        println!(
            "{t:<26} n={n:<3} memory F1 {:.2}  baseline F1 {:.2}  EM {em}/{n} vs {eb}/{n}",
            m / *n as f64,
            b / *n as f64
        );
        n_all += n;
        mf += m;
        bf += b;
        me += em;
        be += eb;
    }
    if n_all > 0 {
        println!(
            "OVERALL                   n={n_all:<3} memory F1 {:.2}  baseline F1 {:.2}  EM {me}/{n_all} vs {be}/{n_all}",
            mf / n_all as f64,
            bf / n_all as f64
        );
    }
    println!("wall: {:.0}s", t0.elapsed().as_secs_f64());
}
