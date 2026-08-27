//! Real-model integration behind the `llm` feature: an OpenAI-compatible
//! client (LM Studio, llama.cpp server, vLLM, OpenAI, ...) with a pluggable
//! transport, plus [`LlmClientExtractor`] and [`HttpEmbedder`] implementing
//! the [`Extractor`](crate::agent::Extractor) and
//! [`Embedder`](crate::semantics::Embedder) traits against a live server.
//!
//! The transport is a function from request JSON to response JSON so tests
//! run without a server; [`OpenAiClient::new`] installs the HTTP transport.

use crate::agent::{CandidateFact, Episode, Extractor};
use crate::semantics::Embedder;
use std::collections::HashMap;
use std::time::Duration;

pub type Transport = Box<dyn Fn(&str) -> Result<String, String> + Send>;

/// Remove `<think>...</think>` blocks reasoning models emit before the
/// answer (also tolerates an unclosed block).
pub fn strip_think(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => {
                rest = &rest[start + end + "</think>".len()..];
            }
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// OpenAI-compatible chat/embeddings client.
pub struct OpenAiClient {
    pub base_url: String,
    pub model: String,
    pub embed_model: String,
    pub temperature: f64,
    pub timeout_secs: u64,
    transport: Transport,
}

impl OpenAiClient {
    /// HTTP transport against `base_url` (e.g. `http://localhost:1234/v1`).
    /// If `LEMMALOG_API_KEY` is set, it is sent as a bearer token (works
    /// with Anthropic's OpenAI-compatible endpoint
    /// `https://api.anthropic.com/v1`, OpenAI, OpenRouter, ...).
    pub fn new(base_url: &str, model: &str, embed_model: &str) -> Self {
        let base = base_url.trim_end_matches('/').to_string();
        let url = base.clone();
        let api_key = std::env::var("LEMMALOG_API_KEY")
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .ok();
        OpenAiClient {
            base_url: base,
            model: model.to_string(),
            embed_model: embed_model.to_string(),
            temperature: 0.0,
            timeout_secs: 120,
            transport: Box::new(move |body: &str| {
                // reasoning models can think for minutes before the answer
                let url = format!("{url}/chat/completions");
                let mut req = ureq::post(&url)
                    .set("Content-Type", "application/json")
                    .timeout(Duration::from_secs(900));
                if let Some(key) = &api_key {
                    // cover both auth styles: OpenAI-compatible (bearer)
                    // and Anthropic native (x-api-key + version header)
                    req = req
                        .set("Authorization", &format!("Bearer {key}"))
                        .set("x-api-key", key)
                        .set("anthropic-version", "2023-06-01");
                }
                match req.send_string(body) {
                    Ok(resp) => resp
                        .into_string()
                        .map_err(|e| format!("read: {e}")),
                    Err(ureq::Error::Status(code, resp)) => {
                        let body = resp
                            .into_string()
                            .unwrap_or_else(|_| String::from("<no body>"));
                        let hint: String = body.chars().take(300).collect();
                        Err(format!("http {code}: {hint}"))
                    }
                    Err(e) => Err(format!("http: {e}")),
                }
            }),
        }
    }

    /// Test transport: no network, deterministic canned responses.
    pub fn with_transport(transport: Transport, model: &str) -> Self {
        OpenAiClient {
            base_url: String::new(),
            model: model.to_string(),
            embed_model: String::new(),
            temperature: 0.0,
            timeout_secs: 120,
            transport,
        }
    }

    /// One chat completion; returns the assistant message content.
    /// Reasoning-model artifacts (`<think>` blocks, LM Studio's
    /// `reasoning_content`) are stripped from the result.
    pub fn chat(&self, system: &str, user: &str) -> Result<String, String> {
        // no temperature field: newer hosted models reject it outright, and
        // protocol-style prompts want deterministic output anyway
        let _ = self.temperature;
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })
        .to_string();
        let resp = (self.transport)(&body)?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| format!("bad json: {e}"))?;
        let msg = &v["choices"][0]["message"];
        let content = msg["content"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .or_else(|| {
                msg["reasoning_content"]
                    .as_str()
                    .map(|s| s.to_string())
            })
            .ok_or_else(|| format!("no content in response: {resp}"))?;
        Ok(strip_think(&content))
    }

    /// One embeddings-endpoint call (static: embeddings use /embeddings,
    /// not the chat transport).
    pub fn http_embed(base_url: &str, embed_model: &str, text: &str) -> Result<Vec<f32>, String> {
        let base = base_url.trim_end_matches('/');
        let url = format!("{base}/embeddings");
        let body = serde_json::json!({"model": embed_model, "input": text}).to_string();
        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(60))
            .send_string(&body)
            .map_err(|e| format!("http: {e}"))?
            .into_string()
            .map_err(|e| format!("read: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| format!("bad json: {e}"))?;
        v["data"][0]["embedding"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                    .collect()
            })
            .ok_or_else(|| format!("no embedding in response: {resp}"))
    }
}

/// [`Extractor`] backed by a live model: prompts with the extraction
/// protocol, parses `S --rel[conf]--> O` lines, memoizes by episode id.
/// Observability: `calls`, `chars_in/out`, `last_latency_ms`.
pub struct LlmClientExtractor {
    client: OpenAiClient,
    prompt: &'static str,
    seen: HashMap<String, Vec<CandidateFact>>,
    pub calls: usize,
    pub failures: usize,
}

impl LlmClientExtractor {
    pub fn new(client: OpenAiClient) -> Self {
        LlmClientExtractor {
            client,
            prompt: crate::agent::EXTRACTION_PROMPT,
            seen: HashMap::new(),
            calls: 0,
            failures: 0,
        }
    }

    /// Override the extraction prompt (e.g. the open-vocabulary variant
    /// used by the LongMemEval runner).
    pub fn with_prompt(mut self, prompt: &'static str) -> Self {
        self.prompt = prompt;
        self
    }

    /// Wrap in a file cache keyed by episode content: extraction results
    /// persist across runs in `dir`, so re-running the same instances pays
    /// only for answering — and with LEMMALOG_NO_ANSWER, context-assembly
    /// iteration costs nothing. Spend once, iterate free.
    pub fn file_cached(self, dir: &str) -> FileCachedExtractor {
        FileCachedExtractor {
            inner: self,
            dir: dir.to_string(),
        }
    }
}

/// [`LlmClientExtractor`] with a persistent file cache: episode text is
/// hashed to a filename holding the extracted triples in line protocol.
/// Hits cost no API calls; misses extract once and write through.
pub struct FileCachedExtractor {
    inner: LlmClientExtractor,
    dir: String,
}

fn cache_key(text: &str, ts: i64) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    ts.hash(&mut h);
    format!("{:016x}", h.finish())
}

impl Extractor for FileCachedExtractor {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        let key = cache_key(&episode.text, episode.ts);
        let path = std::path::Path::new(&self.dir).join(&key);
        if let Ok(cached) = std::fs::read_to_string(&path) {
            return crate::agent::parse_protocol_strict(&cached, 0.9);
        }
        let facts = self.inner.extract(episode);
        let _ = std::fs::create_dir_all(&self.dir);
        let body: String = facts
            .iter()
            .map(|f| {
                format!(
                    "{} --{}[{}]--> {}\n",
                    f.subj, f.pred, f.confidence, f.obj
                )
            })
            .collect();
        let _ = std::fs::write(&path, body);
        facts
    }

    fn stats(&self) -> (usize, usize) {
        self.inner.stats()
    }
}

/// Long episodes are split into message-boundary windows of roughly this
/// size: one extraction call per window finds conversational-buried facts
/// a single pass misses.
pub const EXTRACT_CHUNK_TARGET: usize = 3500;

/// Split an episode's text into windows at message boundaries
/// (lines starting with `user:` / `assistant:`), each <= target chars
/// where the episode allows it.
fn chunk_episode(text: &str, target: usize) -> Vec<String> {
    if text.len() <= target {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        let starts_msg = line.starts_with("user:") || line.starts_with("assistant:")
            || line.starts_with("Session ");
        if starts_msg && !cur.is_empty() && cur.len() + line.len() > target {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

impl Extractor for LlmClientExtractor {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        if let Some(cached) = self.seen.get(&episode.id) {
            return cached.clone();
        }
        // role-aware resolution: user lines resolve to the named speaker,
        // assistant lines resolve to the_assistant (so the assistant's
        // "I recommend X" is not rewritten into the user's voice)
        let who = episode
            .speaker
            .as_deref()
            .map(|s| {
                format!(
                    "The user speaks in lines starting 'user:' and is {s}: resolve the \
user's first-person references to {s}. The assistant speaks in lines \
starting 'assistant:': resolve the assistant's first-person references to \
the_assistant, and extract the assistant's recommendations as \
the_assistant --recommended--> Object.\n"
                )
            })
            .unwrap_or_default();
        let chunks = chunk_episode(&episode.text, EXTRACT_CHUNK_TARGET);
        let n = chunks.len();
        let mut out: Vec<CandidateFact> = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let part = if n > 1 {
                format!(" (part {} of {} of one session)", i + 1, n)
            } else {
                String::new()
            };
            let prompt = format!(
                "{who}Episode (timestamp {}){part}:\n{}\n\nAnswer with the triples only.",
                episode.ts, chunk
            );
            self.calls += 1;
            match self.client.chat(self.prompt, &prompt) {
                Ok(response) => {
                    for f in crate::agent::parse_protocol_strict(&response, 0.9) {
                        if !out.contains(&f) {
                            out.push(f);
                        }
                    }
                }
                Err(_) => {
                    self.failures += 1;
                }
            }
        }
        self.seen.insert(episode.id.clone(), out.clone());
        out
    }

    fn stats(&self) -> (usize, usize) {
        (self.calls, self.failures)
    }
}

/// [`Embedder`] backed by an embeddings endpoint (LM Studio, OpenAI, ...).
/// Embeddings are memoized per input text — entities register once, queries
/// hit the cache on repeats.
pub struct HttpEmbedder {
    pub base_url: String,
    pub model: String,
    cache: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    pub calls: std::cell::Cell<usize>,
}

impl HttpEmbedder {
    pub fn new(base_url: &str, model: &str) -> Self {
        HttpEmbedder {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            cache: std::cell::RefCell::new(HashMap::new()),
            calls: std::cell::Cell::new(0),
        }
    }
}

impl Embedder for HttpEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        if let Some(v) = self.cache.borrow().get(text) {
            return v.clone();
        }
        self.calls.set(self.calls.get() + 1);
        let v = OpenAiClient::http_embed(&self.base_url, &self.model, text)
            .unwrap_or_default();
        self.cache.borrow_mut().insert(text.to_string(), v.clone());
        v
    }
}

pub const RULE_WRITING_PROMPT: &str = "\
You write Datalog rules for the Lemmalog engine. Syntax, exactly:\n\
- Variables start uppercase (X, Person); constants are \"quoted strings\" or integers; _ is a wildcard.\n\
- A rule: name: head(X, Y) :- atom(X, Y), other(Y, Z), X \\= Z.\n\
  The name: prefix is optional. The rule ends with a period.\n\
- Comparisons: < =< > >= = \\= ; integer arithmetic on the right side: D = Dm + 1.\n\
- Negation-as-absence: !atom(X) — only over base (non-derived) predicates.\n\
- now(T) binds the current clock.\n\
- Safety: every variable in the head must appear in a positive body atom.\n\
Output ONLY the rules, one per line, no prose, no markdown fences.\n\
Known predicates and sample facts follow.\n";

/// Parse model-authored rule text into validated clauses: each non-empty
/// line must parse as a clause, be range-restricted (head variables bound
/// by a positive body atom), and the whole set must stratify. Returns the
/// clause text lines worth installing — invalid lines are reported, not
/// silently dropped.
pub fn parse_rule_candidates(
    text: &str,
) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim().trim_matches('`');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("```") {
            // stray fence line
            if rest.is_empty() {
                continue;
            }
        }
        let clause = line.trim_end_matches('.').to_string();
        if clause.is_empty() {
            continue;
        }

        // validate: parses, and is range-restricted
        let parsed = crate::ast::parse_program(&format!("{clause}."))
            .map_err(|e| format!("line {clause:?}: {e}"))?;
        if parsed.len() != 1 {
            return Err(format!("line {clause:?}: expected exactly one clause"));
        }
        let c = &parsed[0];
        if !c.is_fact {
            let head_vars: std::collections::BTreeSet<&str> = c
                .head
                .args
                .iter()
                .filter_map(|t| match t {
                    crate::intern::Term::Var(v) => Some(v.as_str()),
                    _ => None,
                })
                .collect();
            let bound: std::collections::BTreeSet<&str> = c
                .body
                .iter()
                .filter_map(|l| match l {
                    crate::ast::Lit::Pos(a) => Some(a),
                    _ => None,
                })
                .flat_map(|a| a.args.iter())
                .filter_map(|t| match t {
                    crate::intern::Term::Var(v) => Some(v.as_str()),
                    _ => None,
                })
                .collect();
            let unbound: Vec<&str> = head_vars.difference(&bound).copied().collect();
            if !unbound.is_empty() {
                return Err(format!(
                    "line {clause:?}: unsafe rule, head variables not bound by a positive body atom: {unbound:?}"
                ));
            }
        }
        out.push(format!("{clause}."));
    }
    if out.is_empty() {
        return Err("no rules found in model output".to_string());
    }
    Ok(out)
}

impl OpenAiClient {
    /// Ask a live model to author rules: pass the target memory's
    /// `Engine::schema_summary()` and rule source text; the response must
    /// pass [`parse_rule_candidates`]. The caller installs the returned
    /// text (e.g. via `AgentMemory::install_rules`), which re-checks
    /// stratification and backfills against the existing store.
    pub fn synthesize_rules(
        &self,
        schema_summary: &str,
        existing_rules: &str,
        request: &str,
    ) -> Result<String, String> {
        let user = format!(
            "{schema_summary}\nExisting rules:\n{existing_rules}\n\nRequest: {request}\n\nRules only:"
        );
        self.chat(RULE_WRITING_PROMPT, &user)
    }
}
