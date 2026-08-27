//! Differential correctness testing: random stratified programs run through
//! the Engine (seminaive + deltas + magic machinery) and compared against a
//! naive brute-force fixpoint oracle over the ground active domain. This is
//! the classic validation technique for Datalog engines: any disagreement
//! between the optimized evaluator and the dead-simple oracle is a bug.

use lemmalog::{Ann, Engine, Value};
use std::collections::BTreeSet;

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
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const CONSTS: [&str; 3] = ["a", "b", "c"];
const EDB: [&str; 2] = ["e0", "e1"];
const IDB: [&str; 2] = ["p0", "p1"];

/// One generated atom: predicate + arg pattern over vars X, Y.
#[derive(Clone)]
struct GenAtom {
    pred: String,
    neg: bool,
    x_is_var: bool,
    y_is_var: bool,
    x_swap: bool, // use Y in first position (shared-var patterns)
}

impl GenAtom {
    fn render(&self, second: &str) -> String {
        let x = if self.x_is_var {
            if self.x_swap { second.to_string() } else { "X".to_string() }
        } else {
            CONSTS[0].to_string()
        };
        let y = if self.y_is_var { second.to_string() } else { CONSTS[1].to_string() };
        format!("{}{}({}, {})", if self.neg { "!" } else { "" }, self.pred, x, y)
    }
}

/// Generate one random program: rules text + EDB facts. IDB rules are
/// range-restricted (head vars occur in a positive body atom) and negation
/// targets EDB predicates only, so programs are stratified by construction.
fn gen_program(rng: &mut Rng) -> (String, Vec<(String, String, String)>) {
    let mut rules = String::new();
    for p in IDB {
        let n_rules = 1 + rng.below(2);
        for _ in 0..n_rules {
            let n_body = 1 + rng.below(2);
            let mut body = Vec::new();
            let mut first_pos: Option<GenAtom> = None;
            for _ in 0..n_body {
                let mut atom = GenAtom {
                    pred: if rng.below(2) == 0 {
                        EDB[rng.below(2)].to_string()
                    } else {
                        IDB[rng.below(2)].to_string()
                    },
                    neg: false,
                    x_is_var: rng.below(4) > 0,
                    y_is_var: rng.below(4) > 0,
                    x_swap: rng.below(4) == 0,
                };
                if atom.pred.starts_with('e') && rng.below(5) == 0 {
                    // negation only against EDB, and only with constant
                    // args: negated-literal variables must be bound by a
                    // positive literal (the standard safety condition)
                    atom.neg = true;
                    atom.x_is_var = false;
                    atom.y_is_var = false;
                    atom.x_swap = false;
                }
                if !atom.neg && first_pos.is_none() {
                    first_pos = Some(atom.clone());
                }
                body.push(atom);
            }
            // ensure at least one positive atom so heads are range-restricted
            if first_pos.is_none() {
                let atom = GenAtom {
                    pred: EDB[rng.below(2)].to_string(),
                    neg: false,
                    x_is_var: true,
                    y_is_var: true,
                    x_swap: false,
                };
                body.insert(0, atom.clone());
                first_pos = Some(atom);
            }
            let rendered: Vec<String> = body.iter().map(|a| a.render("Y")).collect();
            // range-restricted head: a var may appear only if some positive
            // body atom mentions it (unsafe rules are rejected by design)
            let x_used = rendered.iter().any(|a| !a.starts_with('!') && arg_has(a, "X"));
            let y_used = rendered.iter().any(|a| !a.starts_with('!') && arg_has(a, "Y"));
            let head = format!(
                "{}({}, {})",
                p,
                if x_used && rng.below(4) > 0 { "X" } else { CONSTS[rng.below(3)] },
                if y_used && rng.below(4) > 0 { "Y" } else { CONSTS[rng.below(3)] }
            );
            rules.push_str(&format!("{} :- {}.\n", head, rendered.join(", ")));
        }
    }
    // random EDB facts over the constant domain
    let mut facts = Vec::new();
    for p in EDB {
        for _ in 0..4 {
            let (s, o) = (CONSTS[rng.below(3)], CONSTS[rng.below(3)]);
            facts.push((p.to_string(), s.to_string(), o.to_string()));
        }
    }
    (rules, facts)
}

/// Does an atom's rendered text use var `v` in either argument?
fn arg_has(atom: &str, v: &str) -> bool {
    let inner = &atom[atom.find('(').unwrap() + 1..atom.find(')').unwrap()];
    inner.split(',').any(|a| a.trim() == v)
}

/// Naive fixpoint oracle: ground substitutions over {a,b,c}, iterate rules
/// until no new facts. Dead simple by design.
fn naive_fixpoint(rules_text: &str, edb: &[(String, String, String)]) -> BTreeSet<(String, String, String)> {
    // parse rules ourselves (minimal, matching the generator's shapes)
    struct NRule {
        head: (String, String, String), // pred, arg1, arg2 (var names or consts)
        body: Vec<(String, String, String, bool)>, // pred, a1, a2, neg
    }
    let mut rules = Vec::new();
    for line in rules_text.lines() {
        let Some((h, b)) = line.trim().trim_end_matches('.').split_once(":-") else {
            continue;
        };
        let parse_atom = |s: &str| -> (String, String, String) {
            let s = s.trim().trim_start_matches('!');
            let open = s.find('(').unwrap();
            let close = s.find(')').unwrap();
            let pred = s[..open].to_string();
            let args: Vec<&str> = s[open + 1..close].split(',').map(|x| x.trim()).collect();
            (pred, args[0].to_string(), args[1].to_string())
        };
        let negated = |s: &str| s.trim().starts_with('!');
        let head = parse_atom(h);
        // split the body on atom boundaries: ')' followed by ", "
        let mut body = Vec::new();
        let mut cur = String::new();
        let mut chars = b.chars().peekable();
        while let Some(c) = chars.next() {
            cur.push(c);
            if c == ')' && chars.peek() == Some(&',') {
                chars.next(); // ','
                chars.next(); // ' '
                let a = cur.trim().to_string();
                cur.clear();
                let (p, x, y) = parse_atom(&a);
                body.push((p, x, y, negated(&a)));
            }
        }
        if !cur.trim().is_empty() {
            let a = cur.trim().to_string();
            let (p, x, y) = parse_atom(&a);
            body.push((p, x, y, negated(&a)));
        }
        rules.push(NRule { head, body });
    }

    let edb_set: BTreeSet<(String, String, String)> = edb.iter().cloned().collect();
    let mut all: BTreeSet<(String, String, String)> = edb_set.clone();
    loop {
        let mut added = false;
        for r in &rules {
            // ground the body vars over the domain
            let domain: Vec<&String> = all
                .iter()
                .map(|(_, s, _)| s)
                .chain(all.iter().map(|(_, _, o)| o))
                .collect::<Vec<_>>();
            let dom: Vec<String> = if domain.is_empty() {
                CONSTS.iter().map(|s| s.to_string()).collect()
            } else {
                let mut d: BTreeSet<String> = CONSTS.iter().map(|s| s.to_string()).collect();
                d.extend(domain.into_iter().cloned());
                d.into_iter().collect()
            };
            for v1 in &dom {
                for v2 in &dom {
                    let resolve = |a: &str| -> String {
                        match a {
                            "X" => v1.clone(),
                            "Y" => v2.clone(),
                            c => c.to_string(),
                        }
                    };
                    let mut ok = true;
                    for (p, a1, a2, neg) in &r.body {
                        let f = (p.clone(), resolve(a1), resolve(a2));
                        let present = all.contains(&f);
                        if *neg == present {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        let h = (
                            r.head.0.clone(),
                            resolve(&r.head.1),
                            resolve(&r.head.2),
                        );
                        if !all.contains(&h) {
                            all.insert(h);
                            added = true;
                        }
                    }
                }
            }
        }
        if !added {
            break;
        }
    }
    all.into_iter()
        .filter(|(p, _, _)| IDB.contains(&p.as_str()))
        .collect()
}

fn engine_fixpoint(rules_text: &str, edb: &[(String, String, String)]) -> BTreeSet<(String, String, String)> {
    let mut e = Engine::new();
    e.install_program(rules_text).unwrap();
    for (p, s, o) in edb {
        let (sv, ov) = (e.sym(s), e.sym(o));
        e.declare(p, &[sv, ov], Ann::unit());
    }
    e.run();
    let mut out = BTreeSet::new();
    for p in IDB {
        for (k, _) in e.query(p, &[]) {
            out.insert((
                p.to_string(),
                e.interner.display(&k[0]).to_string(),
                e.interner.display(&k[1]).to_string(),
            ));
        }
    }
    out
}

#[test]
fn engine_agrees_with_naive_oracle_on_random_programs() {
    let mut mismatches = 0;
    for seed in 1..=300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        let (rules, facts) = gen_program(&mut rng);
        let want = naive_fixpoint(&rules, &facts);
        let got = engine_fixpoint(&rules, &facts);
        if want != got {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("=== seed {seed} mismatch ===\nrules:\n{rules}\nfacts: {facts:?}\noracle-only: {:?}\nengine-only: {:?}",
                    want.difference(&got).collect::<Vec<_>>(),
                    got.difference(&want).collect::<Vec<_>>());
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/300 random programs disagree with the oracle");
}

#[test]
fn incremental_agrees_with_from_scratch() {
    // same programs, but feed EDB facts one batch at a time with interleaved
    // runs — the incremental path must equal a single-shot evaluation
    for seed in 1..=150u64 {
        let mut rng = Rng(seed.wrapping_mul(0xD1B54A32D192ED03) | 1);
        let (rules, facts) = gen_program(&mut rng);
        let from_scratch = engine_fixpoint(&rules, &facts);

        let mut e = Engine::new();
        e.install_program(&rules).unwrap();
        let mut i = 0;
        for f in facts.iter() {
            let (sv, ov) = (e.sym(&f.1), e.sym(&f.2));
            e.declare(&f.0, &[sv, ov], Ann::unit());
            i += 1;
            if i % 2 == 0 {
                e.run();
            }
        }
        e.run();
        let mut got = BTreeSet::new();
        for p in IDB {
            for (k, _) in e.query(p, &[]) {
                got.insert((
                    p.to_string(),
                    e.interner.display(&k[0]).to_string(),
                    e.interner.display(&k[1]).to_string(),
                ));
            }
        }
        if got != from_scratch {
            eprintln!("=== seed {seed} incremental divergence ===\nrules:\n{rules}\nfacts: {facts:?}\nincremental-only: {:?}\nscratch-only: {:?}",
                got.difference(&from_scratch).collect::<Vec<_>>(),
                from_scratch.difference(&got).collect::<Vec<_>>());
            assert_eq!(got, from_scratch, "seed {seed}");
        }
    }
}

#[test]
fn parser_rejects_garbage_without_panicking() {
    let mut rng = Rng(0xDEADBEEF | 1);
    let alphabet: [&str; 14] = [
        ":-", "(", ")", ".", "!", "X", "p", "\"s\"", "1", ",", "<", "\\=", "+", "_",
    ];
    for _ in 0..2000 {
        let n = 1 + rng.below(12);
        let junk: String = (0..n)
            .map(|_| alphabet[rng.below(alphabet.len())])
            .collect::<Vec<_>>()
            .join(" ");
        // must return Ok or Err, never panic
        let _ = lemmalog::parse_program(&junk);
        let mut e = Engine::new();
        let _ = e.install_program(&junk);
    }
}

// silence dead-code warning for Value import used in signatures
#[allow(dead_code)]
fn _v(_: Value) {}
