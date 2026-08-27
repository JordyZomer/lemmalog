//! A full agent session on AgentMemory: observe -> update policy ->
//! escalation -> maintain -> ask -> context assembly -> why.

use lemmalog::{AgentMemory, MockExtractor};

fn main() {
    let mut m = AgentMemory::new(
        MockExtractor::new(0.9),
        // caller rules compose with the built-in temporal projection
        "reports_to(X,Y) :- current(X,\"manager\",Y).\n\
         trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).\n\
         project(P,Pr) :- current(P,\"works_on\",Pr).\n\
         inherited: project(P,Pr) :- project(Q,Pr), reports_to(P,Q).",
    )
    .unwrap();

    println!("== t=100: first session ==");
    let r = m.observe_at("alice --manager--> bob\nbob --manager--> carol\nalice --works_at--> acme", 100);
    println!("added={}, escalations={}", r.added, r.escalations.len());
    println!("maintain derived {} facts", m.maintain(100));

    println!("\n== t=200: new session, employer changes ==");
    let r = m.observe_at("alice --works_at--> gigant", 200);
    println!("updated={}, escalations={}", r.updated, r.escalations.len());
    println!("maintain re-derived {} facts", m.maintain(200));
    println!("ask current(alice, works_at, O) -> {:?}", m.ask("current(\"alice\", \"works_at\", O)").unwrap());

    println!("\n== t=300: ambiguous conflict escalates ==");
    let r = m.observe_at("alice --likes--> bob\nalice --likes--> carol", 300);
    println!("added={}, escalations={:?}", r.added, r.escalations);
    m.maintain(300);
    println!("escalation queue: {:?}", m.escalations());
    m.resolve_escalation(0); // agent resolved it out-of-band

    println!("\n== agent queries the memory (datalog-as-tool) ==");
    for q in [
        "reports_to(\"alice\", Y)",
        "current(\"bob\", \"works_at\", O)",
        "reports_to(X, \"carol\")",
    ] {
        println!("?- {q}  ==>  {:?}", m.ask(q).unwrap());
    }

    println!("\n== lookahead: what would follow if carol joined under alice? ==");
    let (rows, added) = m.what_if("carol --manager--> alice", "reports_to(\"carol\", Y)").unwrap();
    println!("hypothetical reports_to(carol, Y) => {rows:?} (would add {added} facts)");
    assert!(m.ask("reports_to(\"carol\", Y)").unwrap().is_empty());
    println!("(store untouched: carol still unknown: {:?})", m.ask("reports_to(\"carol\", Y)").unwrap());

    println!("\n== agent installs its own rule batch (versioned, revertable) ==");
    let batch = m.install_rules("skip(X,Z) :- reports_to(X,Y), reports_to(Y,Z).").unwrap();
    let _ = m.maintain(300);
    println!("installed batch {batch}: skip-level pairs = {}", m.ask("skip(X, Y)").unwrap().len());
    m.uninstall_rules(&batch);
    let _ = m.maintain(300);
    println!("after uninstall: skip-level pairs = {}", m.ask("skip(X, Y)").unwrap().len());

    println!("\n== demand-driven query (magic sets; store untouched) ==");
    println!("?- reports_to(\"alice\", Y) deep ==> {:?}", m.ask_deep("reports_to(\"alice\", Y)").unwrap());

    println!("\n== assembled context for a query about alice ==");
    println!("{}", m.context(&["alice"], 120));

    println!("== why is alice transitively reporting to carol? ==");
    println!("{}", m.why("reports_to(alice, carol)"));
}
