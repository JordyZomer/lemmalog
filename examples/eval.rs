//! Synthetic long-horizon evaluation: knowledge updates, multi-hop
//! reasoning, conflict abstention, token economics, latency.
//! `cargo run --release --example eval [seed] [turns]`

use lemmalog::run_eval;

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let turns: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let rep = run_eval(seed, 30, 8, turns, 40);

    println!("=== lemmalog synthetic long-horizon eval (seed {seed}, {turns} turns) ===");
    println!(
        "knowledge updates   : {}/{} correct ({} supersessions applied)",
        rep.employment_correct, rep.employment_q, rep.supersessions
    );
    println!(
        "multi-hop reasoning : {}/{} correct (magic-sets ask_deep, {:.1} ms total)",
        rep.multihop_correct, rep.multihop_q, rep.ask_deep_ms
    );
    println!(
        "conflict abstention : {}/{} conflicted people keep ALL open preferences",
        rep.abstain_correct, rep.abstain_people
    );
    println!(
        "overall accuracy    : {:.1}%",
        rep.accuracy() * 100.0
    );
    println!(
        "token economics     : {} ctx tokens vs {} transcript tokens ({:.1}x saving)",
        rep.context_tokens,
        rep.transcript_tokens,
        rep.token_savings()
    );
    println!(
        "maintenance latency : {:.1} ms total ({:.3} ms/turn)",
        rep.maintain_ms,
        rep.maintain_ms / rep.turns as f64
    );
}
