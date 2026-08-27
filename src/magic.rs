//! Magic-sets demand-driven evaluation: answer a point query against the
//! rule set WITHOUT materializing the full fixpoint.
//!
//! Classic adornment + magic-predicate rewriting for our grammar (no
//! functors): given a goal atom like `reports_to("n0", Y)`, each IDB
//! predicate gets an adorned version `p$bf` (b = bound, f = free position)
//! restricted by a magic predicate `_magic_p$bf` holding only the bindings
//! reachable from the goal. Bottom-up seminaive over the rewritten program
//! then derives only the goal-relevant slice.
//!
//! SIPS: left-to-right binding passing. Head-bound variables and constants
//! make body-atom positions bound; an atom binds all of its variables once
//! evaluated. Comparisons bind nothing (conservative); `now(T)` binds T.
//! Negation is supported only against EDB predicates (no rules) — negation
//! of a derived predicate inside a demand path is rejected.

use crate::ast::{Clause, Lit};
use crate::intern::Term;
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct MagicError(pub String);
impl std::fmt::Display for MagicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "magic-sets error: {}", self.0)
    }
}
impl std::error::Error for MagicError {}

pub struct DemandProgram {
    pub clauses: Vec<Clause>,
    pub answer_pred: String,
}

fn goal_pred_arity(clauses: &[Clause], pred: &str) -> usize {
    clauses
        .iter()
        .find(|c| c.head.pred == pred)
        .map(|c| c.head.args.len())
        .unwrap_or(0)
}

fn adorn_name(pred: &str, adornment: &str) -> String {
    format!("{pred}${adornment}")
}

fn magic_name(pred: &str, adornment: &str) -> String {
    format!("_magic_{pred}${adornment}")
}

fn is_idb(clauses: &[Clause], pred: &str) -> bool {
    clauses.iter().any(|c| c.head.pred == pred)
}

/// Build the demand program for a goal. `materialized` lists predicates
/// whose base relations already hold evaluated facts: all-free adornments
/// of such predicates alias the base relation (a single rule) instead of
/// re-deriving an unrestricted closure — the classic ground-goal blowup
/// through transitive rules.
pub fn build(
    clauses: &[Clause],
    goal: &crate::ast::Atom,
    materialized: &std::collections::BTreeSet<String>,
) -> Result<DemandProgram, MagicError> {
    let goal_adornment: String = goal
        .args
        .iter()
        .map(|t| match t {
            Term::Sym(_) | Term::Int(_) => 'b',
            _ => 'f',
        })
        .collect();

    // worklist of (pred, adornment) to expand
    let mut scheduled: BTreeSet<(String, String)> = BTreeSet::new();
    let mut out: Vec<Clause> = Vec::new();
    let root = (goal.pred.clone(), goal_adornment.clone());
    scheduled.insert(root.clone());
    let mut worklist = vec![root];

    while let Some((pred, adornment)) = worklist.pop() {
        if !adornment.contains('b') && materialized.contains(pred.as_str()) {
            // all-free adornment over a materialized predicate: alias it
            let args: Vec<Term> = (0..goal_pred_arity(clauses, &pred))
                .map(|i| Term::Var(format!("A{i}")))
                .collect();
            out.push(Clause {
                name: None,
                head: crate::ast::Atom {
                    pred: adorn_name(&pred, &adornment),
                    args: args.clone(),
                },
                body: vec![Lit::Pos(crate::ast::Atom { pred: pred.clone(), args })],
                is_fact: false,
            });
            continue;
        }
        for c in clauses.iter().filter(|c| c.head.pred == pred) {
            if c.is_fact {
                // EDB-in-program fact: duplicate under the adorned name
                out.push(Clause {
                    name: None,
                    head: crate::ast::Atom {
                        pred: adorn_name(&pred, &adornment),
                        args: c.head.args.clone(),
                    },
                    body: Vec::new(),
                    is_fact: true,
                });
                continue;
            }
            expand_rule(clauses, c, &pred, &adornment, &mut out, &mut worklist, &mut scheduled)?;
        }
    }

    // seed: the goal's magic fact (ground by construction at 'b' positions)
    let bound_terms: Vec<Term> = goal
        .args
        .iter()
        .zip(goal_adornment.chars())
        .filter(|(_, a)| *a == 'b')
        .map(|(t, _)| t.clone())
        .collect();
    out.push(Clause {
        name: None,
        head: crate::ast::Atom {
            pred: magic_name(&goal.pred, &goal_adornment),
            args: bound_terms,
        },
        body: Vec::new(),
        is_fact: true,
    });

    Ok(DemandProgram {
        clauses: out,
        answer_pred: adorn_name(&goal.pred, &goal_adornment),
    })
}

fn expand_rule(
    clauses: &[Clause],
    c: &Clause,
    pred: &str,
    adornment: &str,
    out: &mut Vec<Clause>,
    worklist: &mut Vec<(String, String)>,
    scheduled: &mut BTreeSet<(String, String)>,
) -> Result<(), MagicError> {
    // variables bound before the body starts: those in bound head positions
    let mut bound_vars: BTreeSet<String> = c
        .head
        .args
        .iter()
        .zip(adornment.chars())
        .filter(|(_, a)| *a == 'b')
        .filter_map(|(t, _)| match t {
            Term::Var(v) => Some(v.clone()),
            _ => None,
        })
        .collect();
    let head_bound_terms: Vec<Term> = c
        .head
        .args
        .iter()
        .zip(adornment.chars())
        .filter(|(_, a)| *a == 'b')
        .map(|(t, _)| t.clone())
        .collect();

    // pass 1: adorn body atoms left-to-right (SIPS)
    // pos_info: (original pred, adornment, bound args, index in lits)
    let mut lits: Vec<Lit> = Vec::new();
    let mut pos_info: Vec<(String, String, Vec<Term>, usize)> = Vec::new();
    for lit in &c.body {
        match lit {
            Lit::Pos(atom) => {
                let mut adorned_pred = atom.pred.clone();
                if is_idb(clauses, &atom.pred) {
                    let ad: String = atom
                        .args
                        .iter()
                        .map(|t| match t {
                            Term::Sym(_) | Term::Int(_) | Term::Agg(..) => 'b',
                            Term::Var(v) => {
                                if bound_vars.contains(v) {
                                    'b'
                                } else {
                                    'f'
                                }
                            }
                            Term::Wildcard => 'f',
                        })
                        .collect();
                    let bound_args: Vec<Term> = atom
                        .args
                        .iter()
                        .zip(ad.chars())
                        .filter(|(_, a)| *a == 'b')
                        .map(|(t, _)| t.clone())
                        .collect();
                    let key = (atom.pred.clone(), ad.clone());
                    if scheduled.insert(key.clone()) {
                        worklist.push(key);
                    }
                    pos_info.push((atom.pred.clone(), ad, bound_args, lits.len()));
                    adorned_pred = adorn_name(&atom.pred, &pos_info.last().unwrap().1);
                }
                lits.push(Lit::Pos(crate::ast::Atom {
                    pred: adorned_pred,
                    args: atom.args.clone(),
                }));
                for t in &atom.args {
                    if let Term::Var(v) = t {
                        bound_vars.insert(v.clone());
                    }
                }
            }
            Lit::Neg(atom) => {
                if is_idb(clauses, &atom.pred) {
                    return Err(MagicError(format!(
                        "negation of derived predicate {} inside a demand path is unsupported",
                        atom.pred
                    )));
                }
                lits.push(lit.clone());
            }
            Lit::Cmp(..) => {
                lits.push(lit.clone()); // binds nothing (conservative)
            }
            Lit::Now(t) => {
                if let Term::Var(v) = t {
                    bound_vars.insert(v.clone());
                }
                lits.push(lit.clone());
            }
        }
    }

    // pass 2: the adorned rule itself
    let mut rule_body = vec![Lit::Pos(crate::ast::Atom {
        pred: magic_name(pred, adornment),
        args: head_bound_terms.clone(),
    })];
    rule_body.extend(lits.iter().cloned());
    out.push(Clause {
        name: c.name.clone(),
        head: crate::ast::Atom {
            pred: adorn_name(pred, adornment),
            args: c.head.args.clone(),
        },
        body: rule_body,
        is_fact: false,
    });

    // pass 3: magic rules for each adorned body atom
    for (p, ad, bound_args, idx) in &pos_info {
        let mut mbody = vec![Lit::Pos(crate::ast::Atom {
            pred: magic_name(pred, adornment),
            args: head_bound_terms.clone(),
        })];
        mbody.extend(lits[..*idx].to_vec());
        out.push(Clause {
            name: None,
            head: crate::ast::Atom {
                pred: magic_name(p, ad),
                args: bound_args.clone(),
            },
            body: mbody,
            is_fact: false,
        });
    }
    Ok(())
}
