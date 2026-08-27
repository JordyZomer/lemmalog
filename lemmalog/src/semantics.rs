//! Semantic side index: entity embeddings for seeding the relevance
//! (`near`) diffusion rules — the "hybrid vector + symbolic" half of the
//! design. The embedder is pluggable ([`Embedder`]); [`HashEmbedder`] is a
//! deterministic hashing-trick stand-in for tests and offline use. In
//! production, plug a real embedding model behind the same trait — nothing
//! else changes.

use crate::agent::AgentMemory;
use crate::eval::Ann;
use crate::intern::Value;

/// Pluggable text embedder. Implementations must be deterministic for a
/// given input (embeddings are registered once and compared by cosine).
pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Deterministic hashing-trick embeddings: bag-of-tokens projected onto
/// `dim` buckets by FNV-1a, L2-normalized. No model, no network — good
/// enough to test the retrieval wiring; swap for a real embedder in
/// production.
pub struct HashEmbedder {
    pub dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        HashEmbedder { dim }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let mut tokens = 0usize;
        for tok in text.split(|c: char| !c.is_alphanumeric()) {
            if tok.is_empty() {
                continue;
            }
            let h = fnv1a(tok.to_lowercase().as_bytes());
            v[(h % self.dim as u64) as usize] += 1.0;
            tokens += 1;
        }
        if tokens > 0 {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
        }
        v
    }
}

/// Public re-export for callers (similarity gating in reconciliation).
pub fn cosine_pub(a: &[f32], b: &[f32]) -> f32 {
    cosine(a, b)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = (na * nb).sqrt();
    if d > 0.0 {
        dot / d
    } else {
        0.0
    }
}

/// Entity registry with embeddings: the vector half of hybrid retrieval.
/// Symbolic half is the engine's `near` diffusion over `links` edges.
pub struct SemanticIndex {
    embedder: Box<dyn Embedder>,
    entries: Vec<(String, Vec<f32>)>,
}

impl SemanticIndex {
    pub fn new<E: Embedder + 'static>(embedder: E) -> Self {
        SemanticIndex {
            embedder: Box::new(embedder),
            entries: Vec::new(),
        }
    }

    /// Register (or refresh) an entity with a descriptive profile.
    pub fn register(&mut self, entity: &str, profile: &str) {
        let v = self.embedder.embed(profile);
        match self.entries.iter_mut().find(|(e, _)| e == entity) {
            Some(slot) => slot.1 = v,
            None => self.entries.push((entity.to_string(), v)),
        }
    }

    /// Top-k entities by cosine similarity to the query.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        let q = self.embedder.embed(query);
        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|(e, v)| (e.clone(), cosine(&q, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Relevance rules seeded by `mentions`: bounded PPR-style diffusion where
/// confidence (product t-norm) gives decay for free. Requires `links`
/// facts (e.g. `acme --links--> gigant`) — declare them or extract them
/// like any other relation.
pub const RELEVANCE_RULES: &str = "\
near(S,E,1) :- mentions(S,E).
diffuse: near(S,E2,D) :- near(S,E1,Dm), Dm < 3, D = Dm + 1, current(E1,\"links\",E2).
";

impl<X: crate::agent::Extractor> AgentMemory<X> {
    /// Seed `mentions(S, entity)` facts from the semantic index for a
    /// query, with confidence = cosine similarity. Run `maintain()` after
    /// to diffuse relevance over the graph. Returns the seeded entities.
    pub fn seed_mentions(
        &mut self,
        index: &SemanticIndex,
        session: &str,
        query: &str,
        k: usize,
    ) -> Vec<(String, f32)> {
        let hits = index.search(query, k);
        let s: Value = self.engine.sym(session);
        for (entity, sim) in &hits {
            let e = self.engine.sym(entity);
            self.engine
                .declare("mentions", &[s, e], Ann::base(*sim as f64, ["semantic"]));
        }
        hits
    }
}
