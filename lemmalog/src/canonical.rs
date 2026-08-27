//! Entity resolution by canonicalization: the LLM proposes star-shaped
//! `alias(Local, Canonical)` edges (confidence-tagged), Datalog derives the
//! symmetric-transitive closure, and canonical views project facts through
//! the mapping — read-side only, so retracting a bad alias rebuilds the
//! closure and every downstream view in the same epoch.
//!
//! Three safety properties (see the design discussion):
//! 1. **Star shape, not free-form equivalence**: each local name points at
//!    a declared canonical. Topology violations (a local with two
//!    canonicals; a name that is both local and canonical) derive
//!    `alias_conflict` facts instead of merging identities — surface them
//!    to the agent rather than silently unifying.
//! 2. **Read-side views**: raw facts are never rewritten; canonical
//!    projections are derived relations.
//! 3. **Confidence propagates**: alias edges carry semiring confidence, so
//!    a two-hop `same_as` carries the product of the path — weak merges
//!    are visibly low-confidence in queries and `why()` trees.

use crate::eval::{Ann, Engine};
use crate::intern::Value;

/// The canonicalization rule batch. `entity/1` seeds reflexivity; views
/// join on the DIRECTIONAL `maps_to` (local -> canonical, reflexive
/// fallback) so raw facts project to exactly one canonical spelling —
/// the symmetric `same_as` closure stays available for sameness queries.
pub const CANONICAL_RULES: &str = "\
same_as(X, X) :- entity(X).\n\
same_as(X, Y) :- alias(X, Y).\n\
same_as(X, Y) :- alias(Y, X).\n\
same_as(X, Z) :- same_as(X, Y), same_as(Y, Z).\n\
aliased(X) :- alias(X, _).\n\
maps_to(X, X) :- entity(X), !aliased(X).\n\
maps_to(L, C) :- alias(L, C).\n\
alias_conflict(L) :- alias(L, C1), alias(L, C2), C1 \\= C2.\n\
alias_conflict(N) :- alias(N, _), alias(_, N).\n";

impl Engine {
    /// Declare `entity(N)` for every symbol appearing in the given
    /// predicates' rows — the reflexive domain for `same_as`.
    pub fn seed_entities(&mut self, preds: &[&str]) -> usize {
        let mut names: Vec<String> = Vec::new();
        for p in preds {
            for key in self.relation_keys(p) {
                for v in &key {
                    if let Value::Sym(s) = v {
                        let n = self.interner.resolve(*s).to_string();
                        if !names.contains(&n) {
                            names.push(n);
                        }
                    }
                }
            }
        }
        let mut n = 0usize;
        for name in names {
            let sym = self.sym(&name);
            if self.declare("entity", &[sym], Ann::unit()) {
                n += 1;
            }
        }
        n
    }
}

/// Generate a canonical view rule for a relation: `{rel}_canon(...)` with
/// every symbol-typed position projected through `same_as`. Arity 3 is
/// assumed to be (subject, relation-name, object); arity 2 (subject,
/// object); other arities project every position.
pub fn canonical_view_rule(rel: &str, arity: usize) -> String {
    let (raw_args, canon_args): (Vec<String>, Vec<String>) = (0..arity)
        .map(|i| (format!("A{i}"), format!("B{i}")))
        .unzip();
    let mut body = vec![format!("{rel}({})", raw_args.join(", "))];
    for i in 0..arity {
        // arity-3 middle position is the relation name: keep it verbatim
        let skip = arity == 3 && i == 1;
        if !skip {
            // directional: local -> canonical, with reflexive fallback
            body.push(format!("maps_to(A{i}, B{i})"));
        }
    }
    let head = if arity == 3 {
        format!("{rel}_canon({}, A1, {})", canon_args[0], canon_args[2])
    } else {
        format!("{rel}_canon({})", canon_args.join(", "))
    };
    format!("{head} :- {}.\n", body.join(", "))
}

/// Assert one alias edge with confidence (star-shaped: local -> canonical).
pub fn assert_alias(e: &mut Engine, local: &str, canonical: &str, conf: f64) -> bool {
    let (l, c) = (e.sym(local), e.sym(canonical));
    e.declare("alias", &[l, c], Ann::base(conf, ["reconcile"]))
}

/// Install the canonicalization batch, seed the entity domain from the
/// given predicates, and install canonical views for them. Returns the
/// batch id for the rule install.
pub fn install_canonicalization(
    e: &mut Engine,
    preds: &[&str],
) -> Result<String, Box<dyn std::error::Error>> {
    let batch = e.install_program(CANONICAL_RULES)?;
    e.seed_entities(preds);
    let mut views = String::new();
    let mut arities: Vec<(String, usize)> = Vec::new();
    for p in preds {
        if let Some(rel) = e.relations.get(*p) {
            if let Some(a) = rel.rows.first().map(|r| r.key.len()) {
                arities.push((p.to_string(), a));
            }
        }
    }
    for (p, a) in &arities {
        views.push_str(&canonical_view_rule(p, *a));
    }
    if !views.is_empty() {
        e.install_program(&views)?;
    }
    Ok(batch)
}

/// Current conflicts (topology violations to surface, not merge).
pub fn alias_conflicts(e: &Engine) -> Vec<String> {
    e.relation_keys("alias_conflict")
        .into_iter()
        .map(|k| e.render_fact("alias_conflict", &k))
        .collect()
}

#[cfg(feature = "llm")]
pub mod reconcile {
    use super::*;
    use crate::llm::OpenAiClient;
    use crate::llm::HttpEmbedder;
    use crate::semantics::Embedder;

    const RECONCILE_PROMPT: &str = "\
You reconcile entity names in a knowledge graph. You are given candidate \
pairs of names that MIGHT refer to the same real entity (a shorter or \
messier local name vs a fuller canonical name). For each pair you are \
confident refers to the same entity, output one line:\n\
local --alias_of[CONFIDENCE]--> canonical\n\
CONFIDENCE in [0,1] (how sure the two names are the same entity). Choose \
the fuller, cleaner name as the canonical. Skip pairs you are unsure \
about. Output only the lines.";

    /// One reconciliation pass: collect unique entity names, gate
    /// candidate pairs by embedding similarity when an embedder base is
    /// given (otherwise offer all pairs up to a cap), ask the model, and
    /// assert confidence-tagged alias edges. Returns the asserted aliases.
    pub fn reconcile_entities(
        e: &mut Engine,
        chat: &OpenAiClient,
        embed_base: Option<&str>,
        preds: &[&str],
    ) -> Result<Vec<(String, String, f64)>, String> {
        // collect unique entity names
        let mut names: Vec<String> = Vec::new();
        for p in preds {
            for key in e.relation_keys(p) {
                for v in &key {
                    if let Value::Sym(s) = v {
                        let n = e.interner.resolve(*s).to_string();
                        if !names.contains(&n) {
                            names.push(n);
                        }
                    }
                }
            }
        }
        if names.len() < 2 {
            return Ok(Vec::new());
        }
        // candidate pairs: similarity-gated when possible
        let pairs: Vec<(String, String)> = match embed_base {
            Some(base) => {
                let embedder = HttpEmbedder::new(base, "text-embedding-nomic-embed-text-v1.5");
                let mut gated = Vec::new();
                for i in 0..names.len() {
                    for j in i + 1..names.len() {
                        let a = embedder.embed(&names[i]);
                        let b = embedder.embed(&names[j]);
                        let cos = crate::semantics::cosine_pub(&a, &b);
                        if cos > 0.72 {
                            gated.push((names[i].clone(), names[j].clone()));
                        }
                    }
                }
                gated
            }
            None => {
                let mut all = Vec::new();
                for i in 0..names.len() {
                    for j in i + 1..names.len() {
                        all.push((names[i].clone(), names[j].clone()));
                    }
                }
                all.truncate(60);
                all
            }
        };
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let listing = pairs
            .iter()
            .map(|(a, b)| format!("- {a} | {b}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = chat
            .chat(
                RECONCILE_PROMPT,
                &format!("Candidate pairs:\n{listing}\n\nLines only:"),
            )
            .map_err(|e| e.to_string())?;
        let mut asserted = Vec::new();
        for f in crate::agent::parse_protocol_strict(&out, 0.8) {
            if f.pred == "alias_of" {
                assert_alias(e, &f.subj, &f.obj, f.confidence);
                asserted.push((f.subj, f.obj, f.confidence));
            }
        }
        Ok(asserted)
    }
}
