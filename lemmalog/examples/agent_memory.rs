//! End-to-end demo of Lemmalog as an agent memory:
//! ingestion -> derivation -> supersession -> query -> why.

use lemmalog::{Ann, Engine, Value};

/// Ingest an episode: the extraction boundary (here a stand-in for the
/// LLM OpenIE step) asserts annotated bi-temporal facts.
fn ingest_episode(
    e: &mut Engine,
    episode: &str,
    ts: i64,
    facts: &[(&str, &str, &str)],
) {
    for (s, p, o) in facts {
        let mut args = vec![e.sym(s), e.sym(p), e.sym(o)];
        args.extend([Value::Int(ts), Value::Int(i64::MAX), Value::Int(ts)]);
        println!("  [ingest/{episode}] {s} --{p}--> {o}");
        e.declare("edge", &args, Ann::base(0.9, [episode]));
    }
}

fn main() {
    let mut e = Engine::new();

    // The rule layer: derived relations ARE the memory.
    e.install_program(
        "# temporal projection: what is true NOW\n\
         current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.\n\
         # transitive closure over the manager chain\n\
         reports_to(X,Y) :- current(X,\"manager\",Y).\n\
         trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).\n\
         # contradiction candidates via a curated exclusivity table\n\
         exclusive(\"works_at\").\n\
         conflict(E,R,O1,O2,T1,T2) :- edge(E,R,O1,_,_,T1), edge(E,R,O2,_,_,T2),\n\
                                      exclusive(R), T2 > T1, O1 \\= O2.\n\
         # HippoRAG-style bounded relevance diffusion\n\
         near(S,X,1) :- mentions(S,X).\n\
         diffuse: near(S,Y,D) :- near(S,X,Dm), Dm < 3, D = Dm + 1, current(X,\"links\",Y).",
    )
    .unwrap();

    println!("== turn 1 (t=100): episode ep1 ==");
    ingest_episode(&mut e, "ep1", 100, &[
        ("alice", "manager", "bob"),
        ("bob", "manager", "carol"),
        ("alice", "works_at", "acme"),
    ]);
    let (s1, acme) = (e.sym("s1"), e.sym("acme"));
    e.declare("mentions", &[s1, acme], Ann::base(0.9, ["q1"]));
    e.set_now(100);
    println!("derived {} facts", e.run());
    println!("reports_to(alice, *):");
    let a1 = e.sym("alice");
    for (k, _) in e.query("reports_to", &[Some(a1), None]) {
        println!("  {}", e.render_fact("reports_to", &k));
    }

    println!("\n== why does alice report to carol? ==");
    let (alice, carol) = (e.sym("alice"), e.sym("carol"));
    println!("{}", e.why("reports_to", &[alice, carol]));

    println!("\n== turn 2 (t=200): episode ep2 supersedes alice's employer ==");
    ingest_episode(&mut e, "ep2", 200, &[("alice", "works_at", "gigant")]);
    // close the old edge's valid_to: retraction-by-annotation
    let mut old = vec![e.sym("alice"), e.sym("works_at"), e.sym("acme")];
    old.extend([Value::Int(100), Value::Int(i64::MAX), Value::Int(100)]);
    e.retract("edge", &old);
    let mut closed = old.clone();
    closed[4] = Value::Int(200);
    e.declare("edge", &closed, Ann::base(0.9, ["ep1"]));
    e.set_now(200);
    println!("re-derived {} facts after supersession", e.run());
    let (a3, wa) = (e.sym("alice"), e.sym("works_at"));
    for (k, a) in e.query("current", &[Some(a3), Some(wa), None]) {
        println!("current employer: {} (conf {:.2})", e.render_fact("current", &k), a.conf);
    }
    println!("conflicts flagged: {}", e.query("conflict", &[]).len());

    println!("\n== turn 3 (t=300): nothing new -> zero work ==");
    println!("derived {} facts", e.run());

    println!("\n== relevance diffusion from a query mentioning acme ==");
    let (s2, acme2) = (e.sym("s2"), e.sym("acme"));
    e.declare("mentions", &[s2, acme2], Ann::base(0.8, ["q2"]));
    ingest_episode(&mut e, "ep3", 300, &[("acme", "links", "gigant"), ("gigant", "links", "zeta")]);
    e.set_now(300);
    e.run();
    let s2q = e.sym("s2");
    for (k, a) in e.query("near", &[Some(s2q), None, None]) {
        println!("near: {} (conf {:.3})", e.render_fact("near", &k), a.conf);
    }
}
