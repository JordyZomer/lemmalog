use lemmalog::{Ann, Engine, Value};

fn syms(e: &mut Engine, xs: &[&str]) -> Vec<Value> {
    xs.iter().map(|x| e.sym(x)).collect()
}

#[test]
fn count_min_max_sum() {
    let mut e = Engine::new();
    e.install_program(
        "kit_count(P, count(K)) :- bought(P, K).\n\
         kit_max(P, max(N)) :- rating(P, N).\n\
         kit_min(P, min(N)) :- rating(P, N).\n\
         total(P, sum(N)) :- rating(P, N).",
    )
    .unwrap();
    let rows = [("alice", "f15"), ("alice", "spitfire"), ("alice", "tiger"), ("bob", "camaro")];
    for (p, k) in rows {
        let v = syms(&mut e, &[p, k]);
        e.declare("bought", &v, Ann::unit());
    }
    for (p, n) in [("alice", 5), ("alice", 3), ("alice", 4), ("bob", 2)] {
        let p = e.sym(p);
        e.declare("rating", &[p, Value::Int(n)], Ann::unit());
    }
    e.run();
    let alice = e.sym("alice");
    let c = e.query("kit_count", &[Some(alice), None]);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].0[1], Value::Int(3), "count distinct kits");
    assert_eq!(e.query("kit_max", &[Some(alice), None])[0].0[1], Value::Int(5));
    assert_eq!(e.query("kit_min", &[Some(alice), None])[0].0[1], Value::Int(3));
    assert_eq!(e.query("total", &[Some(alice), None])[0].0[1], Value::Int(12));
    // bob's group separate
    let bob = e.sym("bob");
    assert_eq!(e.query("kit_count", &[Some(bob), None])[0].0[1], Value::Int(1));
}

#[test]
fn count_grows_and_propagates() {
    let mut e = Engine::new();
    e.install_program(
        "kit_count(P, count(K)) :- bought(P, K).\n\
         big_spender(P) :- kit_count(P, N), N >= 3.",
    )
    .unwrap();
    for k in ["a", "b"] {
        let v = syms(&mut e, &["alice", k]);
        e.declare("bought", &v, Ann::unit());
    }
    e.run();
    assert_eq!(e.query("kit_count", &[]).len(), 1);
    assert_eq!(e.query("kit_count", &[None, None])[0].0[1], Value::Int(2));
    assert_eq!(e.query("big_spender", &[]).len(), 0);

    // third kit arrives in a later epoch: count 2 -> 3 flips the reader
    let v = syms(&mut e, &["alice", "c"]);
    e.declare("bought", &v, Ann::unit());
    e.run();
    let rows = e.query("kit_count", &[]);
    assert_eq!(rows.len(), 1, "no stale count row");
    assert_eq!(rows[0].0[1], Value::Int(3));
    assert_eq!(e.query("big_spender", &[]).len(), 1, "value change propagates");
}

#[test]
fn retraction_shrinks_count() {
    let mut e = Engine::new();
    e.install_program("kit_count(P, count(K)) :- bought(P, K).").unwrap();
    let kits: Vec<Vec<Value>> = ["a", "b", "c"]
        .iter()
        .map(|k| syms(&mut e, &["alice", k]))
        .collect();
    for k in &kits {
        e.declare("bought", k, Ann::unit());
    }
    e.run();
    assert_eq!(e.query("kit_count", &[])[0].0[1], Value::Int(3));
    assert!(e.retract("bought", &kits[0]));
    e.run();
    let rows = e.query("kit_count", &[]);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0[1], Value::Int(2), "count recomputed after retraction");
}

#[test]
fn multiple_aggregate_columns() {
    let mut e = Engine::new();
    e.install_program("stats(P, count(K), max(R)) :- bought(P, K, R).").unwrap();
    for (k, r) in [("a", 3), ("b", 7), ("c", 5)] {
        let alice = e.sym("alice");
        let v = vec![alice, e.sym(k), Value::Int(r)];
        e.declare("bought", &v, Ann::unit());
    }
    e.run();
    let rows = e.query("stats", &[]);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0[1], Value::Int(3));
    assert_eq!(rows[0].0[2], Value::Int(7));
}

#[test]
fn rejects_bad_aggregation_programs() {
    // aggregate in a body atom
    let mut e = Engine::new();
    assert!(e
        .install_program("p(X) :- q(X), r(count(X)).")
        .is_err());
    // recursion through the aggregated head
    let mut e = Engine::new();
    assert!(e
        .install_program("p(X, count(Y)) :- p(X, Y), q(X, Y).")
        .is_err(),
        "aggregation over its own head must be rejected");
    // mixed definition
    let mut e = Engine::new();
    assert!(e
        .install_program("p(X, count(Y)) :- q(X, Y).\np(X, Y) :- q(X, Y).")
        .is_err());
    // unbound group variable
    let mut e = Engine::new();
    assert!(e.install_program("p(Z, count(Y)) :- q(X, Y).").is_err());
}

#[test]
fn aggregate_witness_in_why() {
    let mut e = Engine::new();
    e.install_program("kits: kit_count(P, count(K)) :- bought(P, K).").unwrap();
    let v = syms(&mut e, &["alice", "spitfire"]);
    e.declare("bought", &v, Ann::unit());
    e.run();
    let alice = e.sym("alice");
    let out = e.why("kit_count", &[alice, Value::Int(1)]);
    assert!(out.contains("via kits"), "{out}");
}
