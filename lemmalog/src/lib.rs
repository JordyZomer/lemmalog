//! # Lemmalog
//!
//! A Datalog engine for LLM context management: bi-temporal,
//! provenance-annotated, incrementally maintained memory for agents.
//!
//! The model:
//! - **Base facts** (`edge`, `mentions`, ...) are asserted by the ingestion
//!   layer with confidence and provenance (source episode ids).
//! - **Rules** are runtime-installed, stratified Datalog. Derived relations
//!   (closures, temporal projection, contradiction candidates, relevance
//!   diffusion) are the *memory*.
//! - Every fact — asserted or derived — carries a semiring annotation;
//!   confidence multiplies across rule bodies (product t-norm), provenance
//!   unions across rule bodies.
//! - `why()` renders the proof tree of any fact back to source episodes.
//! - `run()` is incremental: only rules reachable from newly asserted facts
//!   re-fire (seminaive over per-epoch deltas).
//!
//! Example rule program (the temporal-projection + closure pattern from the
//! design doc):
//!
//! ```text
//! current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.
//! reports_to(X,Y) :- edge(X,"manager",Y,_,_,_).
//! reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).
//! ```

pub mod agent;
pub mod canonical;
pub mod ast;
pub mod eval;
#[cfg(feature = "llm")]
pub mod llm;
#[cfg(feature = "llm")]
pub mod longmemeval;
pub mod intern;
pub mod magic;
pub mod retrieval;
pub mod scenario;
pub mod semantics;
pub mod session;

pub use agent::{
    assemble_context, AgentMemory, Episode, Extractor, IngestReport, LlmExtractor, MockExtractor,
    DEFAULT_RULES, EXTRACTION_PROMPT,
};
pub use scenario::{run_eval, EvalReport, Scenario};
pub use retrieval::{Bm25, Retrieval, Selection};
pub use semantics::{Embedder, HashEmbedder, SemanticIndex, RELEVANCE_RULES};
pub use ast::{parse_program, ParseError};
pub use eval::{Ann, Change, Engine, StoredFact, StratError};
pub use intern::{Interner, Term, Value};

impl Engine {
    /// Install (append) a rule program. Rules are identified by optional
    /// `name:` prefixes; unnamed rules get `rule/<head-pred>` labels used in
    /// `why()` output.
    /// Install (append) a rule program as a versioned batch; returns the
    /// batch id for later `uninstall`. Installing marks the program dirty:
    /// the next `run()` backfills every rule against the existing store
    /// (not just newly asserted facts).
    pub fn install_program(&mut self, src: &str) -> Result<String, Box<dyn std::error::Error>> {
        let clauses = parse_program(src)?;
        self.validate_aggregates(&clauses)?;
        for (i, c) in clauses.iter().enumerate() {
            if !c.is_fact {
                self.ever_derived.insert(c.head.pred.clone());
                if c.head.args.iter().any(|t| matches!(t, Term::Agg(..))) {
                    // the lowered temp relation is derived state too
                    let temp_ci = self.clauses.len() + i;
                    self.ever_derived
                        .insert(format!("__agg:{}:{temp_ci}", c.head.pred));
                }
            }
        }
        self.clauses.extend(clauses);
        self.check_program()?;
        let id = format!("b{}", self.rule_batches.len());
        self.rule_batches
            .push((id.clone(), src.to_string(), self.clauses.len()));
        self.program_dirty = true;
        Ok(id)
    }

    /// Validate aggregation clauses before installation: Agg terms only
    /// in head arguments; group and inner variables bound by positive
    /// body atoms (range restriction).
    fn validate_aggregates(
        &self,
        clauses: &[crate::ast::Clause],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::ast::Lit;
        fn term_has_agg(t: &Term) -> bool {
            match t {
                Term::Agg(..) => true,
                _ => false,
            }
        }
        for c in clauses {
            if c.is_fact {
                continue;
            }
            // binding is only REQUIRED for aggregation clauses (their fold
            // evaluates head terms from body solutions); ordinary clauses
            // may bind heads via comparisons (D = Dm + 1)
            let is_agg = c.head.args.iter().any(|t| matches!(t, Term::Agg(..)));
            let mut bound: std::collections::BTreeSet<&str> = Default::default();
            let mut agg_in_body = false;
            for lit in &c.body {
                match lit {
                    Lit::Pos(a) | Lit::Neg(a) => {
                        for t in &a.args {
                            if term_has_agg(t) {
                                agg_in_body = true;
                            }
                            if let Term::Var(v) = t {
                                if matches!(lit, Lit::Pos(_)) {
                                    bound.insert(v.as_str());
                                }
                            }
                        }
                    }
                    Lit::Cmp(_, t, e) => {
                        let mut check = |t: &Term| {
                            if term_has_agg(t) {
                                agg_in_body = true;
                            }
                        };
                        check(t);
                        check_expr(e, &mut check);
                    }
                    Lit::Now(t) => {
                        if term_has_agg(t) {
                            agg_in_body = true;
                        }
                    }
                }
            }
            if agg_in_body {
                return Err(format!(
                    "aggregates are only allowed in rule heads: {:?}",
                    c.head.pred
                )
                .into());
            }
            if !is_agg {
                continue;
            }
            let mut head_vars: Vec<&str> = Vec::new();
            for t in &c.head.args {
                match t {
                    Term::Var(v) => head_vars.push(v.as_str()),
                    Term::Agg(_, inner) => {
                        if let Term::Var(v) = &**inner {
                            head_vars.push(v.as_str());
                        }
                    }
                    _ => {}
                }
            }
            let unbound: Vec<&str> = head_vars
                .iter()
                .filter(|v| !bound.contains(*v))
                .copied()
                .collect();
            if !unbound.is_empty() {
                return Err(format!(
                    "unsafe aggregation rule {:?}: head variables {unbound:?} not bound by a positive body atom",
                    c.head.pred
                )
                .into());
            }
        }
        Ok(())
    }
}

fn check_expr<F: FnMut(&Term)>(e: &crate::ast::Expr, f: &mut F) {
    use crate::ast::Expr;
    match e {
        Expr::T(t) => f(t),
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            check_expr(a, f);
            check_expr(b, f);
        }
    }
}
