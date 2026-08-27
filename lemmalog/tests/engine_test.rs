use lemmalog::{Ann, Engine, Value};

fn syms(e: &mut Engine, xs: &[&str]) -> Vec<Value> {
    xs.iter().map(|x| e.sym(x)).collect()
}

fn edge(e: &mut Engine, subj: &str, pred: &str, obj: &str, vf: i64, vt: i64, ts: i64) {
    let mut args = syms(e, &[subj, pred, obj]);
    args.extend([Value::Int(vf), Value::Int(vt), Value::Int(ts)]);
    e.declare("edge", &args, Ann::unit());
}

#[test]
fn transitive_closure() {
    let mut e = Engine::new();
    e.install_program(
        "reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).\n\
         reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    )
    .unwrap();
    edge(&mut e, "alice", "manager", "bob", 0, i64::MAX, 1);
    edge(&mut e, "bob", "manager", "carol", 0, i64::MAX, 1);
    edge(&mut e, "carol", "manager", "dana", 0, i64::MAX, 1);
    e.run();
    let alice = e.sym("alice");
    let res = e.query("reports_to", &[Some(alice), None]);
    assert_eq!(res.len(), 3, "transitive closure should derive 3 facts");
}

#[test]
fn temporal_projection_and_updates() {
    let mut e = Engine::new();
    e.install_program(
        "current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.",
    )
    .unwrap();
    edge(&mut e, "alice", "works_at", "acme", 0, i64::MAX, 1);
    e.set_now(10);
    e.run();
    assert_eq!(e.query("current", &[]).len(), 1);

    // supersede: invalidate the old edge at t=20, assert the new one
    let old = {
        let mut v = syms(&mut e, &["alice", "works_at", "acme"]);
        v.extend([Value::Int(0), Value::Int(i64::MAX), Value::Int(1)]);
        v
    };
    assert!(e.retract("edge", &old));
    edge(&mut e, "alice", "works_at", "acme", 0, 20, 1);
    edge(&mut e, "alice", "works_at", "gigant", 20, i64::MAX, 2);
    e.set_now(25);
    e.run();
    let alice = e.sym("alice");
    let cur = e.query("current", &[Some(alice), None, None]);
    assert_eq!(cur.len(), 1);
    assert_eq!(cur[0].0[2], e.sym("gigant"));
}

#[test]
fn confidence_propagates_as_product() {
    let mut e = Engine::new();
    e.install_program("b(X) :- a(X). c(X) :- b(X).").unwrap();
    let x = e.sym("x");
    e.declare("a", &[x], Ann::base(0.5, ["ep1"]));
    e.run();
    let c = e.query("c", &[Some(x)]);
    assert_eq!(c.len(), 1);
    assert!((c[0].1.conf - 0.5).abs() < 1e-9);
    assert!(c[0].1.prov.contains("ep1"));
}

#[test]
fn provenance_union_across_rule_body() {
    let mut e = Engine::new();
    e.install_program("derived(X) :- p(X), q(X).").unwrap();
    let x = e.sym("x");
    e.declare("p", &[x], Ann::base(0.9, ["ep1"]));
    e.declare("q", &[x], Ann::base(0.9, ["ep2"]));
    e.run();
    let d = e.query("derived", &[Some(x)]);
    assert!(d[0].1.prov.contains("ep1") && d[0].1.prov.contains("ep2"));
    assert!((d[0].1.conf - 0.81).abs() < 1e-9);
}

#[test]
fn why_renders_proof_tree() {
    let mut e = Engine::new();
    e.install_program(
        "reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).\n\
         trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    )
    .unwrap();
    edge(&mut e, "a", "manager", "b", 0, i64::MAX, 1);
    edge(&mut e, "b", "manager", "c", 0, i64::MAX, 1);
    // re-assert with provenance-bearing annotations
    let ab = syms(&mut e, &["a", "manager", "b"]);
    let ab_args = [ab[0], ab[1], ab[2], Value::Int(0), Value::Int(i64::MAX), Value::Int(1)];
    e.declare("edge", &ab_args, Ann::base(0.9, ["ep0"]));
    let bc = syms(&mut e, &["b", "manager", "c"]);
    let bc_args = [bc[0], bc[1], bc[2], Value::Int(0), Value::Int(i64::MAX), Value::Int(1)];
    e.declare("edge", &bc_args, Ann::base(0.9, ["ep1"]));
    e.run();
    let (a, c) = (e.sym("a"), e.sym("c"));
    let out = e.why("reports_to", &[a, c]);
    assert!(out.contains("via trans"), "proof tree: {out}");
    assert!(out.contains("ep0") && out.contains("ep1"), "proof tree: {out}");
    assert!(out.contains("asserted (base fact)"), "proof tree: {out}");
}

#[test]
fn incremental_epochs() {
    let mut e = Engine::new();
    e.install_program(
        "reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).\n\
         reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    )
    .unwrap();
    edge(&mut e, "a", "manager", "b", 0, i64::MAX, 1);
    edge(&mut e, "b", "manager", "c", 0, i64::MAX, 1);
    assert_eq!(e.run(), 3, "2 direct + 1 transitive (a->c) reports_to");
    assert_eq!(e.run(), 0, "no new inputs: zero derivations");

    // new edge extends closure incrementally
    edge(&mut e, "c", "manager", "d", 0, i64::MAX, 2);
    assert_eq!(e.run(), 3, "only a->d, b->d, c->d re-derived");
    let (a, d) = (e.sym("a"), e.sym("d"));
    assert_eq!(e.query("reports_to", &[Some(a), Some(d)]).len(), 1);
}

#[test]
fn stratified_negation() {
    let mut e = Engine::new();
    e.install_program(
        "current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.\n\
         orphan(E) :- entity(E), !current(E,_,_).",
    )
    .unwrap();
    let (alice, bob) = (e.sym("alice"), e.sym("bob"));
    e.declare("entity", &[alice], Ann::unit());
    e.declare("entity", &[bob], Ann::unit());
    edge(&mut e, "alice", "works_at", "acme", 0, i64::MAX, 1);
    e.set_now(5);
    e.run();
    let orphans = e.query("orphan", &[None]);
    assert_eq!(orphans.len(), 1);
    assert_eq!(orphans[0].0[0], e.sym("bob"));
}

#[test]
fn negation_cycle_rejected() {
    let mut e = Engine::new();
    let err = e.install_program("a(X) :- b(X), !c(X). b(X) :- a(X). c(X) :- b(X).");
    assert!(err.is_err(), "recursion through negation must be rejected");
}

#[test]
fn wildcards_and_program_facts() {
    let mut e = Engine::new();
    e.install_program(
        "# curated exclusivity table\n\
         exclusive(\"works_at\").\n\
         conflict(E,R,O1,O2) :- edge(E,R,O1,_,_,T1), edge(E,R,O2,_,_,T2), exclusive(R), T2 > T1, O1 \\= O2.",
    )
    .unwrap();
    edge(&mut e, "alice", "works_at", "acme", 0, 20, 1);
    edge(&mut e, "alice", "works_at", "gigant", 20, i64::MAX, 2);
    e.run();
    assert_eq!(e.query("conflict", &[None, None, None, None]).len(), 1);
}

#[test]
fn merge_keeps_max_confidence_and_union_prov() {
    let mut e = Engine::new();
    let x = e.sym("x");
    e.declare("p", &[x], Ann::base(0.5, ["ep1"]));
    e.declare("p", &[x], Ann::base(0.9, ["ep2"]));
    let f = e.fact("p", &[x]).unwrap();
    assert!((f.ann.conf - 0.9).abs() < 1e-9);
    assert!(f.ann.prov.contains("ep1") && f.ann.prov.contains("ep2"));
    // two independent derivations of the same fact record both supports
    e.install_program("q(X) :- p(X).\n alt: q(X) :- p(X), p(X).")
        .unwrap();
    e.run();
    let q = e.fact("q", &[x]).unwrap();
    assert_eq!(q.supports.len(), 2);
}

#[test]
fn bounded_relevance_diffusion() {
    // HippoRAG-style salience diffusion: product t-norm gives decay for free
    let mut e = Engine::new();
    e.install_program(
        "near(S,E,1) :- mentions(S,E).\n\
         near(S,E2,D) :- near(S,E1,Dm), Dm < 3, D = Dm + 1, edge2(E1,E2).",
    )
    .unwrap();
    let sess = e.sym("sess1");
    let (a, b, c) = (e.sym("a"), e.sym("b"), e.sym("c"));
    e.declare("mentions", &[sess, a], Ann::base(0.9, ["q"]));
    e.declare("edge2", &[a, b], Ann::base(0.8, ["kg"]));
    e.declare("edge2", &[b, c], Ann::base(0.5, ["kg"]));
    e.run();
    let b_facts = e.query("near", &[Some(sess), Some(b), None]);
    assert_eq!(b_facts.len(), 1);
    assert!((b_facts[0].1.conf - 0.9 * 0.8).abs() < 1e-9);
    let c_facts = e.query("near", &[Some(sess), Some(c), None]);
    assert_eq!(c_facts.len(), 1);
    assert!((c_facts[0].1.conf - 0.9 * 0.8 * 0.5).abs() < 1e-9);
    // depth bound holds
    let d = e.sym("d");
    e.declare("edge2", &[c, d], Ann::base(0.9, ["kg"]));
    e.run();
    assert_eq!(e.query("near", &[Some(sess), Some(d), None]).len(), 0);
}

#[test]
fn scoped_recompute_on_retract() {
    let mut e = Engine::new();
    e.install_program(
        "reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).\n\
         reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).\n\
         unrelated(X) :- other(X).",
    )
    .unwrap();
    edge(&mut e, "a", "manager", "b", 0, i64::MAX, 1);
    edge(&mut e, "b", "manager", "c", 0, i64::MAX, 1);
    let z = e.sym("z");
    e.declare("other", &[z], Ann::unit());
    assert_eq!(e.run(), 4); // (a,b),(b,c),(a,c) + unrelated(z)
    assert_eq!(e.query("unrelated", &[]).len(), 1);

    // retract b->c: only reports_to recomputes; unrelated is untouched
    let bc = {
        let mut v = syms(&mut e, &["b", "manager", "c"]);
        v.extend([Value::Int(0), Value::Int(i64::MAX), Value::Int(1)]);
        v
    };
    assert!(e.retract("edge", &bc));
    // run() counts NET NEW facts: (a,b) survived the rebuild (existed
    // before and after), (b,c)/(a,c) were dropped -> zero new facts
    let n = e.run();
    assert_eq!(n, 0, "no net-new facts; two stale derivations dropped");
    assert_eq!(e.query("reports_to", &[]).len(), 1);
    assert_eq!(e.query("reports_to", &[]).len(), 1);
    assert_eq!(e.query("unrelated", &[]).len(), 1, "unrelated survives recompute");
}

#[test]
fn ask_deep_matches_full_fixpoint() {
    // 10 disjoint chains of 10 nodes: full closure = 10 * 45 = 450 facts;
    // a demand query about chain 0 should touch only chain 0's slice.
    let rules = "reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).\n\
         reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).";
    let mut e = Engine::new();
    e.install_program(rules).unwrap();
    let manager = e.sym("manager");
    for c in 0..10 {
        for i in 0..9 {
            let (a, b) = (e.sym(&format!("c{c}n{i}")), e.sym(&format!("c{c}n{}", i+1)));
            e.declare("edge", &[a, manager, b, Value::Int(0), Value::Int(i64::MAX), Value::Int(1)], Ann::unit());
        }
    }
    e.run();
    let full_closure = e.query("reports_to", &[]).len();
    assert_eq!(full_closure, 450);
    let c0n0 = e.sym("c0n0");
    let full: Vec<String> = e
        .query("reports_to", &[Some(c0n0), None])
        .into_iter()
        .map(|(k, _)| e.interner.display(&k[1]).to_string())
        .collect();
    assert_eq!(full.len(), 9);

    // fresh engine, same data, demand evaluation only
    let mut e2 = Engine::new();
    e2.install_program(rules).unwrap();
    let manager2 = e2.sym("manager");
    for c in 0..10 {
        for i in 0..9 {
            let (a, b) = (e2.sym(&format!("c{c}n{i}")), e2.sym(&format!("c{c}n{}", i+1)));
            e2.declare("edge", &[a, manager2, b, Value::Int(0), Value::Int(i64::MAX), Value::Int(1)], Ann::unit());
        }
    }
    let rows = e2.ask_deep("reports_to(\"c0n0\", Y)").unwrap();
    let mut got: Vec<String> = rows
        .iter()
        .map(|r| r.strip_prefix("Y=").unwrap().to_string())
        .collect();
    let mut want = full.clone();
    want.sort();
    got.sort();
    assert_eq!(got, want, "demand answers must equal full fixpoint answers");
    // the demand slice is chain-0 sized, not graph sized
    assert!(
        e2.last_demand_facts < full_closure / 4,
        "demand slice = {} vs full closure = {}",
        e2.last_demand_facts,
        full_closure
    );
    assert_eq!(e2.query("reports_to", &[]).len(), 0, "base store untouched");
}

#[test]
fn ask_deep_no_rewrites_edb() {
    let mut e = Engine::new();
    let x = e.sym("x");
    e.declare("p", &[x], Ann::unit());
    assert_eq!(e.ask_deep("p(X)").unwrap(), vec!["X=x".to_string()]);
}

#[test]
fn ask_deep_through_projection() {
    let mut e = Engine::new();
    e.install_program(
        "current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.\n\
         reports_to(X,Y) :- current(X,\"manager\",Y).\n\
         trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    )
    .unwrap();
    edge(&mut e, "a", "manager", "b", 0, i64::MAX, 1);
    edge(&mut e, "b", "manager", "c", 0, i64::MAX, 1);
    e.set_now(5);
    let rows = e.ask_deep("reports_to(\"a\", Y)").unwrap();
    let got: Vec<&str> = rows.iter().map(|r| r.strip_prefix("Y=").unwrap()).collect();
    assert_eq!(got, vec!["b", "c"]);
}

#[test]
fn retract_middle_row_does_not_corrupt_store() {
    // regression: swap_remove mishandling resurrected the removed fact and
    // silently deleted a neighbor
    let mut e = Engine::new();
    e.install_program("q(X) :- p(X).").unwrap();
    let keys: Vec<Value> = ["a", "b", "c", "d"].iter().map(|s| e.sym(s)).collect();
    for k in &keys {
        e.declare("p", &[*k], Ann::unit());
    }
    e.run();
    assert_eq!(e.query("q", &[]).len(), 4);
    // remove a middle row (not the last) -> relocation path
    assert!(e.retract("p", &[keys[1]]));
    assert_eq!(e.query("p", &[]).len(), 3, "b removed");
    let gone = e.query("p", &[Some(keys[1])]);
    assert!(gone.is_empty(), "b must stay gone");
    // every other fact survives exactly once, including the relocated one
    for i in [0, 2, 3] {
        assert_eq!(e.query("p", &[Some(keys[i])]).len(), 1, "key {i}");
    }
    // derived relations recompute without ghosts or losses
    let n = e.run();
    assert_eq!(e.query("q", &[]).len(), 3);
    let _ = n;
    // repeated retract/declare cycles stay consistent (victims cycle
    // over all four keys, so every fact is eventually removed)
    for round in 0..10 {
        let victim = keys[round % 4];
        e.declare("p", &[victim], Ann::unit());
        assert!(e.retract("p", &[victim]));
    }
    assert_eq!(e.query("p", &[]).len(), 0);
}

#[test]
fn mid_session_rule_install_backfills() {
    // regression: a rule installed after facts existed never fired, because
    // evaluation only reacted to pending deltas
    let mut e = Engine::new();
    edge(&mut e, "a", "manager", "b", 0, i64::MAX, 1);
    e.run();
    assert_eq!(e.query("reports_to", &[]).len(), 0); // no rules yet
    e.install_program("reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).")
        .unwrap();
    let n = e.run();
    assert!(n >= 1, "new rule must backfill against existing store");
    assert_eq!(e.query("reports_to", &[]).len(), 1);
}

#[test]
fn rule_batches_install_and_uninstall() {
    let mut e = Engine::new();
    e.install_program("p(X) :- s(X).").unwrap();
    let b2 = e.install_program("q(X) :- p(X).").unwrap();
    let x = e.sym("x");
    e.declare("s", &[x], Ann::unit());
    e.run();
    assert_eq!(e.query("q", &[]).len(), 1);
    assert_eq!(e.batches().len(), 2);

    // uninstalling the second batch drops its derivations
    assert!(e.uninstall(&b2));
    assert!(!e.uninstall("nonexistent"));
    e.run();
    assert_eq!(e.query("p", &[]).len(), 1, "first batch untouched");
    assert_eq!(e.query("q", &[]).len(), 0, "q derivations reverted");
}

#[test]
fn change_feed_streams_adds_retractions_and_clears() {
    use lemmalog::Change;
    let mut e = Engine::new();
    e.install_program("q(X) :- p(X).").unwrap();
    let (a, b) = (e.sym("a"), e.sym("b"));
    e.declare("p", &[a], Ann::unit());
    e.declare("p", &[b], Ann::unit());
    e.run();
    let after_epoch0 = e.epoch();
    let feed = e.changes_since(after_epoch0);
    assert!(feed.is_empty(), "idle epoch emits nothing");

    // additions stream
    let epoch_before = e.epoch();
    let c = e.sym("c");
    e.declare("p", &[c], Ann::unit());
    e.run();
    let feed = e.changes_since(epoch_before);
    assert!(feed.contains(&Change::Added(1, ("p".into(), vec![c]))));
    assert!(feed.contains(&Change::Added(1, ("q".into(), vec![c]))));

    // explicit retraction streams
    let epoch_before = e.epoch();
    assert!(e.retract("p", &[b]));
    e.run();
    let feed = e.changes_since(epoch_before);
    assert!(
        feed.contains(&Change::Retracted(2, ("p".into(), vec![b]))),
        "{feed:?}"
    );
    // q's rebuild is a wholesale clear of q
    assert!(
        feed.iter().any(|c| matches!(c, Change::Cleared(_, p) if p == "q")),
        "{feed:?}"
    );
    assert_eq!(e.query("q", &[]).len(), 2, "q rebuilt without b");
}

#[test]
fn negation_invalidated_by_later_additions() {
    // regression (found by differential testing): a rule deriving via
    // negation keeps stale facts when the negated predicate grows later
    let mut e = Engine::new();
    e.install_program("p(X) :- e(X), !blocked(X).").unwrap();
    let a = e.sym("a");
    e.declare("e", &[a], Ann::unit());
    e.run();
    assert_eq!(e.query("p", &[]).len(), 1, "not blocked yet");

    // blocking fact arrives in a LATER turn: p(a) must be retracted
    e.declare("blocked", &[a], Ann::unit());
    e.run();
    assert_eq!(e.query("p", &[]).len(), 0, "stale p(a) must not survive");
    // and unblocking (retract) restores it
    assert!(e.retract("blocked", &[a]));
    e.run();
    assert_eq!(e.query("p", &[]).len(), 1);
}
