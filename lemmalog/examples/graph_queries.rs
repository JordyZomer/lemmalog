//! Cyclic join stress: triangle detection and reachability over a random
//! graph — the query class where worst-case-optimal joins (leapfrog
//! triejoins) matter. Per-position hash indexes make the nested-loop
//! seminaive evaluator competitive at agent-memory scale; this example
//! measures where that stops being true.
//!
//! `cargo run --release --example graph_queries [nodes] [avg-degree]`

use lemmalog::{Ann, Engine, Value};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn main() {
    let nodes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let deg: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let edges = nodes * deg;

    let mut e = Engine::new();
    e.install_program(
        "# symmetric closure + triangle detection (cyclic join)\n\
         sym(A,B) :- arc(A,B).\n\
         sym(A,B) :- arc(B,A).\n\
         tri(A,B,C) :- arc(A,B), arc(B,C), arc(C,A).\n\
         reach(A,B) :- arc(A,B).\n\
         step: reach(A,C) :- reach(A,B), arc(B,C).",
    )
    .unwrap();

    let mut rng = Rng(0xC0FFEE | 1);
    let node: Vec<_> = (0..nodes).map(|i| e.sym(&format!("n{i}"))).collect();
    let t0 = Instant::now();
    for _ in 0..edges {
        let a = node[(rng.next() as usize) % nodes];
        let b = node[(rng.next() as usize) % nodes];
        if a != b {
            e.declare(
                "arc",
                &[a, b],
                Ann::unit(),
            );
        }
    }
    let ingest = t0.elapsed();

    let t1 = Instant::now();
    let derived = e.run();
    let eval = t1.elapsed();

    let triangles = e.query("tri", &[]).len();
    let reach = e.query("reach", &[]).len();
    let arcs = e.query("arc", &[]).len();

    let t2 = Instant::now();
    let one = e.ask_deep("reach(\"n0\", \"n7\")").unwrap().len();
    let point = t2.elapsed();

    let t3 = Instant::now();
    let extra = e.sym("x");
    let n0 = e.sym("n0");
    e.declare("arc", &[extra, n0], Ann::unit());
    let inc = e.run();
    let inc_t = t3.elapsed();

    println!("nodes={nodes} deg={deg} arcs={arcs}");
    println!("ingest          : {ingest:?}");
    println!(
        "fixpoint        : {eval:?} (+{derived} facts: {triangles} triangles, {reach} reach)"
    );
    println!("point query     : {point:?} (answer rows: {one})");
    println!("incremental turn: {inc_t:?} (+{inc} facts from one arc)");
}
