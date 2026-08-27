//! Scale sanity check: chain closure + incremental turn + point lookup.
//! Run with `cargo run --release --example perf [n]` (default 500).
//! The initial fixpoint is O(n^3) work for a chain; the point of this
//! example is the incremental/idle turn contrast, not raw fixpoint speed.

use lemmalog::{Ann, Engine, Value};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let mut e = Engine::new();
    e.install_program(
        "reports_to(X,Y) :- edge(X,\"manager\",Y,_,_,_).\n\
         reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    )
    .unwrap();

    // chain n0 -> n1 -> ... -> n{n-1}
    let mut names = Vec::with_capacity(n);
    for i in 0..n {
        names.push(e.sym(&format!("n{i}")));
    }
    let manager = e.sym("manager");
    let t0 = Instant::now();
    for i in 0..n - 1 {
        e.declare(
            "edge",
            &[names[i], manager, names[i + 1], Value::Int(0), Value::Int(i64::MAX), Value::Int(1)],
            Ann::unit(),
        );
    }
    let ingest = t0.elapsed();

    let t1 = Instant::now();
    let derived = e.run();
    let closure_secs = t1.elapsed().as_secs_f64();
    let expected = n * (n - 1) / 2;
    assert_eq!(derived, expected, "closure must be complete");

    // incremental turn: one new edge at the end
    let extra = e.sym("n_extra");
    e.declare(
        "edge",
        &[names[n - 1], manager, extra, Value::Int(0), Value::Int(i64::MAX), Value::Int(2)],
        Ann::unit(),
    );
    let t2 = Instant::now();
    let inc = e.run();
    let inc_secs = t2.elapsed().as_secs_f64();
    assert_eq!(inc, n, "one new edge -> n new closure facts");

    // idle turn
    let t3 = Instant::now();
    let idle = e.run();
    let idle_secs = t3.elapsed().as_secs_f64();
    assert_eq!(idle, 0);

    println!("n                     : {n}");
    println!("ingest {}/{} edges     : {:?}", n - 1, n - 1, ingest);
    println!("closure fixpoint      : {derived} facts in {closure_secs:.3}s");
    println!("incremental turn      : +{inc} facts in {inc_secs:.6}s");
    println!("idle turn             : {idle} facts in {idle_secs:.6}s");
}
