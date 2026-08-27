//! The investigation demo: an agent's beliefs as a derived graph.
//!
//! A security investigation accumulates observations (episodes), and rules
//! derive the consequences: evidence chains, an exploit chain, hypotheses,
//! a decision. Then the foundational observation is retracted — and every
//! conclusion that depended on it ceases to hold, automatically, in the
//! same epoch, while unrelated conclusions keep their proofs.
//!
//! This is the architectural thesis in one run: the agent owns perception
//! (which observations to record, which to retract); the engine owns state
//! and consequence. Vector-memory systems can retrieve the observations;
//! they cannot maintain the consequences.
//!
//!     cargo run --release --example investigation

use lemmalog::agent::{AgentMemory, MockExtractor};
use std::time::Instant;

const RULES: &str = "\
# observations live as edge(S, R, O) triples; view them as predicates\n\
primitive_reachable(P) :- current(_, \"found_primitive\", P).\n\
target_mapping(T) :- current(_, \"maps_target\", T).\n\
target_mapping(T) :- current(_, \"confirms_mapping\", T).\n\
# the exploit chain: reachable primitive + attacker-controlled target\n\
exploit_viable(P) :- primitive_reachable(P), target_mapping(attacker_controlled).\n\
# decisions are derivations, not vibes\n\
decision(escalate_to_ir, P) :- exploit_viable(P).\n\
# hypothesis lifecycle\n\
supported(h_auth_bypass) :- primitive_reachable(write_phys).\n\
refuted(h_benign_flag) :- current(_, \"found\", benign_flag_disabled).\n";

fn main() {
    let mut m = AgentMemory::new(MockExtractor::new(0.9), RULES).unwrap();
    let t0 = Instant::now();

    // ---- the investigation: observations with provenance ----
    m.observe_at("debugger --found_primitive--> write_phys", 100); // ep1
    m.observe_at(
        "source_review --maps_target[0.7]--> attacker_controlled", // ep2 (shaky)
        120,
    );
    m.observe_at(
        "runtime_trace --confirms_mapping--> attacker_controlled", // ep3
        140,
    );
    m.observe_at("code_audit --found--> benign_flag_disabled", 160); // ep4
    m.maintain(160);

    println!("== after 4 observations ({} ms) ==", t0.elapsed().as_millis());
    for pred in ["primitive_reachable", "target_mapping", "exploit_viable"] {
        let rows = m.engine.query(pred, &[]);
        for (k, _) in rows {
            println!("  {}", m.engine.render_fact(pred, &k));
        }
    }
    let h = m.engine.sym("h_auth_bypass");
    let flag = m.engine.sym("h_benign_flag");
    for (k, _) in m.engine.query("supported", &[]) {
        println!("  supported: {}", m.engine.render_fact("supported", &k));
    }
    for (k, _) in m.engine.query("refuted", &[]) {
        println!("  refuted:   {}", m.engine.render_fact("refuted", &k));
    }
    let (escalate, wp) = (m.engine.sym("escalate_to_ir"), m.engine.sym("write_phys"));
    let decisions = m.engine.query("decision", &[Some(escalate), Some(wp)]);
    assert_eq!(decisions.len(), 1, "the decision derives from the evidence");

    // ---- the proof: why do we believe this decision? ----
    println!("\n== why(decision(escalate_to_ir, write_phys)) ==");
    print!("{}", m.engine.why("decision", &[escalate, wp]));

    // ---- the retraction: the target-mapping claim was a misread ----
    println!("\n== retracting the foundational observations ==");
    for (rel_name, ts) in [("maps_target", 120i64), ("confirms_mapping", 140)] {
        let rel = m.engine.sym(rel_name);
        let ac = m.engine.sym("attacker_controlled");
        let rows = m.engine.query(
            "edge",
            &[None, Some(rel), Some(ac), None, None, None],
        );
        for (k, _) in rows {
            if k[5].as_int() == Some(ts) {
                let retracted = m.engine.retract("edge", &k);
                println!(
                    "  retracted ep@{ts} ({}): {}",
                    if retracted { "ok" } else { "MISS" },
                    m.engine.render_fact("edge", &k)
                );
            }
        }
    }
    let derived = m.maintain(200);
    println!("closure repaired: {derived} facts re-derived");

    // ---- the aftermath: what still holds? ----
    println!("\n== after retraction ==");
    let gone = m.engine.query("decision", &[Some(escalate), Some(wp)]).is_empty();
    let viable_gone = m.engine.query("exploit_viable", &[]).is_empty();
    let still_supported = m.engine.query("supported", &[Some(h)]).len() == 1;
    let still_refuted = m.engine.query("refuted", &[Some(flag)]).len() == 1;
    println!(
        "decision(escalate_to_ir):  {}  <-- CEASED TO HOLD",
        if gone { "gone" } else { "STILL PRESENT (bug)" }
    );
    println!(
        "exploit_viable(write_phys): {}",
        if viable_gone { "gone" } else { "STILL PRESENT (bug)" }
    );
    println!(
        "supported(h_auth_bypass):  {}  <-- hypothesis survives: its own evidence is intact",
        if still_supported { "present" } else { "GONE (bug)" }
    );
    println!(
        "refuted(h_benign_flag):    {}  <-- refutation unaffected",
        if still_refuted { "present" } else { "GONE (bug)" }
    );
    println!(
        "\nThe decision was never deleted. It was a CONSEQUENCE, and\n\
consequences are recomputed when the facts change. Total wall time: {} ms.",
        t0.elapsed().as_millis()
    );
}
