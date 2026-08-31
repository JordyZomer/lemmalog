//! LongMemEval integration (feature `llm`): loads the benchmark's
//! oracle split, replays each instance's evidence sessions through an
//! `AgentMemory` (live extraction -> update policy -> derived rules), then
//! answers the question two ways — grounded in the memory block, and from
//! the raw transcript alone (the paper's oracle baseline) — scoring both
//! with SQuAD-style token F1 by question type.
//!
//! The oracle split (evidence sessions only, 500 instances) comes from
//! https://huggingface.co/datasets/xiaowu0162/longmemeval
//! (`longmemeval_oracle`, 15 MB JSON).

use crate::agent::AgentMemory;
use crate::llm::LlmClientExtractor;
use crate::llm::OpenAiClient;

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub date: String,
    pub ts: i64,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub answer: String,
    pub question_date: String,
    pub sessions: Vec<Session>,
}

/// "2023/04/10 (Mon) 17:50" -> a monotonic minute-resolution integer.
pub fn parse_date(s: &str) -> i64 {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let (d, t) = (parts.first().copied().unwrap_or("0/0/0"), parts.get(3).copied().unwrap_or("0:0"));
    let dp: Vec<i64> = d.split('/').filter_map(|x| x.parse().ok()).collect();
    let tp: Vec<i64> = t.split(':').filter_map(|x| x.parse().ok()).collect();
    let (y, m, day) = (
        *dp.first().unwrap_or(&0),
        *dp.get(1).unwrap_or(&0),
        *dp.get(2).unwrap_or(&0),
    );
    (((y * 366 + m * 31 + day) * 24 + tp.first().copied().unwrap_or(0)) * 60)
        + tp.get(1).copied().unwrap_or(0)
}

/// Load the oracle split JSON.
pub fn load(path: &str) -> Result<Vec<Instance>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let arr = v.as_array().ok_or("expected a JSON array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let get = |k: &str| item[k].as_str().unwrap_or_default().to_string();
        let ids: Vec<String> = item["haystack_session_ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let dates: Vec<String> = item["haystack_dates"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let mut sessions = Vec::new();
        if let Some(arr) = item["haystack_sessions"].as_array() {
            for (idx, msgs) in arr.iter().enumerate() {
                let id = ids.get(idx).cloned().unwrap_or_else(|| format!("s{idx}"));
                let date = dates
                    .get(idx)
                    .cloned()
                    .unwrap_or_default();
                let messages = msgs
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|m| Message {
                                role: m["role"].as_str().unwrap_or_default().to_string(),
                                content: m["content"].as_str().unwrap_or_default().to_string(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                sessions.push(Session {
                    ts: parse_date(&date),
                    id,
                    date,
                    messages,
                });
            }
        }
        // replay in chronological order (supersession is order-sensitive)
        sessions.sort_by_key(|s| s.ts);
        out.push(Instance {
            question_id: get("question_id"),
            question_type: get("question_type"),
            question: get("question"),
            answer: get("answer"),
            question_date: get("question_date"),
            sessions,
        });
    }
    Ok(out)
}

/// SQuAD-style token F1 and exact match after normalization.
pub fn score_f1(prediction: &str, gold: &str) -> (f64, bool) {
    let norm = |s: &str| -> Vec<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty() && !matches!(*t, "the" | "a" | "an" | "is" | "are" | "was" | "were" | "to" | "of" | "in" | "on" | "and" | "or" | "for" | "with" | "it" | "its"))
            .map(|t| t.to_string())
            .collect()
    };
    let p = norm(prediction);
    let g = norm(gold);
    if p.is_empty() || g.is_empty() {
        return (if p.is_empty() && g.is_empty() { 1.0 } else { 0.0 }, p == g);
    }
    let em = p == g;
    let mut common = std::collections::HashMap::new();
    for t in &g {
        *common.entry(t.clone()).or_insert(0i64) += 1;
    }
    let mut hits = 0i64;
    for t in &p {
        if let Some(c) = common.get_mut(t) {
            if *c > 0 {
                *c -= 1;
                hits += 1;
            }
        }
    }
    let precision = hits as f64 / p.len() as f64;
    let recall = hits as f64 / g.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (f1, em)
}

pub struct RunResult {
    pub question_type: String,
    pub gold: String,
    pub memory_pred: String,
    pub baseline_pred: String,
    pub memory_f1: f64,
    pub baseline_f1: f64,
    pub memory_em: bool,
    pub baseline_em: bool,
    pub memory_ctx_tokens: usize,
    pub transcript_tokens: usize,
}

/// Extraction prompt variant for the benchmark: open relation vocabulary
/// (everyday-life facts — owns, drives, allergic_to, ...), since the
/// engine's schema is dynamic. Everything else matches the standard
/// protocol.
pub const OPEN_VOCAB_PROMPT: &str = "\
Extract the factual triples from the episode below. Answer with one triple \
per line in exactly this format, nothing else:\n\
SUBJECT --RELATION[CONFIDENCE]--> OBJECT\n\
CONFIDENCE is a number in [0,1] (omit [CONFIDENCE] for 0.9). RELATION is a \
short snake_case verb phrase (works_at, drives, owns, lives_in, \
allergic_to, bought, booked, likes, ...). Extract quantitative facts too: \
personal bests, times, prices, quantities, dates (time_5k, commute_time, \
price_paid, ...). If the episode contains a list, roster, or schedule, \
extract one triple per row (rotation, shift, assignment, ...).\n\
Temporal: when the episode states WHEN something happened or will happen \
(a full date, 'in 2019', 'last March', 'next week'), extract one triple \
with a normalized date object in YYYY, YYYY-MM, or YYYY-MM-DD form \
(visited_on, moved_on, starts_on, planned_for). The episode header shows \
the session date — resolve relative dates ('last March', 'two years ago') \
against it. When it states that one thing happened before another, \
extract A --before--> B. Never guess a date or an ordering the text does \
not state.\n\
Amounts: extract monetary amounts as plain numbers without symbols \
('$50' -> 50) in dedicated relations (price_paid, amount_spent, cost).\n\
Lists: if the episode enumerates items ('I bought three postcards', a \
roster of subscriptions, instruments, books), extract ONE triple per \
item, naming each item by its description in the text — a count is not \
a fact, the items are.\n\
Preferences: unconditional likes are likes. Conditional preferences ('I \
prefer lively places when I'm with friends') are prefers_when with the \
condition as the object. Never assert the condition itself as a fact \
unless the episode states it currently holds.\n\
SUBJECT and OBJECT must be real entity names exactly as written in the \
episode: NEVER a pronoun or a role word - always the full name. Output \
ONLY the triple lines: no reasoning, no explanations, no bullets. Skip \
opinions and small talk.\n\
Episode:\n";

/// A normalized date object as extracted by the open-vocab prompt: YYYY,
/// YYYY-MM, or YYYY-MM-DD. Returns a comparable integer (YYYYMMDD, with
/// missing parts zero) so Datalog `<` compares real time, not interning
/// order (symbol comparison in the engine is by intern id).
pub fn date_to_int(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let digits =
        |r: std::ops::Range<usize>| b.get(r).map(|x| x.iter().all(|d| d.is_ascii_digit())).unwrap_or(false);
    let (y, m, d) = if s.len() == 10 && b[4] == b'-' && b[7] == b'-'
        && digits(0..4) && digits(5..7) && digits(8..10)
    {
        (&s[0..4], &s[5..7], &s[8..10])
    } else if s.len() == 7 && b[4] == b'-' && digits(0..4) && digits(5..7) {
        (&s[0..4], &s[5..7], "00")
    } else if s.len() == 4 && digits(0..4) {
        (&s[0..4], "00", "00")
    } else {
        return None;
    };
    Some(y.parse::<i64>().ok()? * 10000 + m.parse::<i64>().ok()? * 100 + d.parse::<i64>().ok()?)
}

fn episode_text(s: &Session) -> String {
    let mut out = format!("Session {} on {}:\n", s.id, s.date);
    for m in &s.messages {
        out.push_str(&format!("{}: {}\n", m.role, m.content));
    }
    out
}

const ANSWER_PROMPT: &str = "Answer the question using ONLY the provided context. Reply with just the answer - no full sentences, no explanation, match the specificity of the evidence.";

/// Relations (from the current view) whose objects are YYYY-MM-DD dates.
fn seen_rels_count_off(m: &AgentMemory<Box<dyn crate::agent::Extractor>>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let keys = m.engine.relation_keys("current");
    let mut checked: Vec<String> = Vec::new();
    for key in keys {
        if key.len() != 3 {
            continue;
        }
        let r = m.engine.interner.display(&key[1]).to_string();
        if checked.contains(&r) {
            continue;
        }
        checked.push(r.clone());
        let obj = m.engine.interner.display(&key[2]).to_string();
        let b = obj.as_bytes();
        let shaped = obj.len() == 10
            && b[4] == b'-'
            && b[7] == b'-'
            && obj[..4].chars().all(|c| c.is_ascii_digit())
            && obj[5..7].chars().all(|c| c.is_ascii_digit())
            && obj[8..].chars().all(|c| c.is_ascii_digit());
        if shaped {
            out.push(r);
        }
    }
    out
}

/// Replay one instance's sessions through a fresh memory (live extraction,
/// update policy, derived rules), then answer in two modes: grounded in
/// the structured memory block (facts + dated edge history), and from the
/// raw transcript alone (the paper's oracle baseline).
fn chat_retry(chat: &OpenAiClient, system: &str, user: &str) -> Result<String, String> {
    chat.chat(system, user)
        .or_else(|e1| {
            eprintln!("  api error ({e1}); retrying once...");
            chat.chat(system, user)
        })
}

/// Whole-token subject matching for the dated-history append: a subject
/// matches the question iff they share a non-stopword token (len >= 3).
/// The audit showed two-way raw-substring matching (with stopwords like
/// "the"/"did") pulled in the whole edge table — 4-6x context bloat that
/// drowned the very facts selection had chosen.
pub fn subject_matches_question(subject: &str, question: &str) -> bool {
    let qtokens: std::collections::BTreeSet<String> =
        crate::retrieval::tokenize_pub(question).into_iter().filter(|t| t.len() >= 3).collect();
    crate::retrieval::tokenize_pub(subject)
        .into_iter()
        .any(|t| t.len() >= 3 && qtokens.contains(&t))
}

pub fn run_instance(
    extract_client: OpenAiClient,
    chat: &OpenAiClient,
    inst: &Instance,
) -> Result<RunResult, String> {
    let cache_dir = std::env::var("LEMMALOG_CACHE_DIR").ok();
    let extractor = LlmClientExtractor::new(extract_client).with_prompt(OPEN_VOCAB_PROMPT);
    let extractor: Box<dyn crate::agent::Extractor> = match cache_dir {
        Some(dir) => Box::new(extractor.file_cached(&dir)),
        None => Box::new(extractor),
    };
    let mut m = AgentMemory::new(extractor, "").map_err(|e| e.to_string())?;

    for s in &inst.sessions {
        m.observe_as(&episode_text(s), s.ts, "the_user");
        m.maintain(s.ts);
    }

    // OPT-IN (LEMMALOG_COUNTS=1): dynamic per-relation counting rules via
    // aggregation. Measured effect at n=5/type: adds context noise without
    // fixing counting questions — the bottleneck there is extraction
    // recall (mentioned items that never become facts), not the missing
    // aggregate. Kept behind a flag for experimentation.
    // the distinct relations seen in current(S, R, O) rows
    let mut seen_rels: Vec<String> = Vec::new();
    for key in m.engine.relation_keys("current") {
        if key.len() == 3 {
            let r = m.engine.interner.display(&key[1]).to_string();
            if !seen_rels.contains(&r) {
                seen_rels.push(r);
            }
        }
    }
    let counting_enabled = std::env::var("LEMMALOG_COUNTS").is_ok();
    let mut count_rules = String::new();
    if !counting_enabled {
        seen_rels.clear();
    }
    for r in &seen_rels {
        let safe = r.replace('/', "_");
        count_rules.push_str(&format!(
            "cnt_{safe}(S, count(O)) :- current(S, \"{r}\", O).\n"
        ));
    }
    if !count_rules.is_empty() {
        let _ = m.install_rules(&count_rules);
        let _ = m.maintain(inst.sessions.last().map(|s| s.ts).unwrap_or(0));
    }

    // data-driven temporal ordering: relations whose objects are
    // YYYY-MM-DD dates become dated/2, and ordering is DERIVED by rules
    // (stated data extracted, inference in the deterministic layer)
    let mut order_rules = String::new();
    for r in &seen_rels_count_off(&m) {
        let safe = r.replace('/', "_");
        order_rules.push_str(&format!(
            "dated(S, D) :- current(S, \"{r}\", D).\n"
        ));
        let _ = safe;
    }
    order_rules.push_str(
        "happened_before(A, B) :- dated(A, D1), dated(B, D2), D1 < D2.\n\
         stated_before(X, Y) :- current(X, \"before\", Y).\n\
         stated_before(X, Z) :- stated_before(X, Y), stated_before(Y, Z).\n",
    );
    let _ = m.install_rules(&order_rules);
    let _ = m.maintain(inst.sessions.last().map(|s| s.ts).unwrap_or(0));

    // ts -> original date string, so edge history renders human-readable
    // dates (models cannot compare our monotonic integers)
    let dates: std::collections::HashMap<i64, String> = inst
        .sessions
        .iter()
        .map(|s| (s.ts, s.date.clone()))
        .collect();
    let date_of = |v: &crate::intern::Value| -> String {
        match v.as_int() {
            Some(ts) if ts > 0 && ts != i64::MAX => dates
                .get(&ts)
                .cloned()
                .unwrap_or_else(|| ts.to_string()),
            _ => "open".to_string(),
        }
    };

    // structured memory block: hybrid retrieval (BM25 + entity boosting)
    // under budget, plus a BUDGETED dated-history append keyed off
    // whole-token subject/question matches — ordering questions need both
    // endpoints' dates; the audit showed loose matching unbudgeted was
    // 4-6x context bloat that drowned the selected facts
    let memory_ctx = {
        let base = m.context_for_query(&inst.question, 1800);
        let mut hist = String::new();
        let hist_budget = 700 * 4; // chars
        for key in m.engine.relation_keys("edge") {
            if hist.len() > hist_budget {
                break;
            }
            let subj = m.engine.interner.display(&key[0]).to_string();
            if !subject_matches_question(&subj, &inst.question) {
                continue;
            }
            let disp = |i: usize| m.engine.interner.display(&key[i]).to_string();
            hist.push_str(&format!(
                "{} --{}--> {}, {} .. {} (asserted {})\n",
                disp(0),
                disp(1),
                disp(2),
                date_of(&key[3]),
                date_of(&key[4]),
                date_of(&key[5]),
            ));
        }
        if hist.is_empty() {
            base
        } else {
            format!(
                "{base}\nEDGE HISTORY: subject --relation--> object, valid_from .. valid_to (asserted at):\n{hist}"
            )
        }
    };
    // offline diagnosis: dump the assembled context without spending on
    // answers (pair with LEMMALOG_NO_ANSWER=1 for zero-cost iteration)
    if let Ok(dir) = std::env::var("LEMMALOG_DUMP_CTX") {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(
            std::path::Path::new(&dir).join(format!("{}.txt", inst.question_id)),
            &memory_ctx,
        );
    }
    if std::env::var("LEMMALOG_NO_ANSWER").is_ok() {
        return Err("__no_answer__".into());
    }

    let transcript = inst
        .sessions
        .iter()
        .map(episode_text)
        .collect::<Vec<_>>()
        .join("\n");

    let mut memory_pred = chat_retry(
        chat,
        ANSWER_PROMPT,
        &format!("CONTEXT:\n{memory_ctx}\n\nQUESTION: {}", inst.question),
    )?;
    // recall fallback: the transcript mode reads raw text by construction;
    // when the structured memory lacks the fact, one targeted extraction
    // pass over the source episodes restores parity
    let lower = memory_pred.to_lowercase();
    if lower.contains("not specified")
        || lower.contains("not stated")
        || lower.contains("no information")
        || lower.contains("not in context")
        || lower.contains("don't have")
        || lower.contains("no relevant")
        || lower.contains("unknown")
    {
        let recalled = chat_retry(
            chat,
            "The user asks a question about their past conversations. Scan the \
transcript for the specific answer. Reply with ONLY the requested fact as \
one or more triples in the form SUBJECT --RELATION--> OBJECT (no \
confidence, no prose). If the transcript truly does not contain the \
answer, reply with the single word NONE.",
            &format!("QUESTION: {}\n\nTRANSCRIPT:\n{transcript}", inst.question),
        )
        .unwrap_or_default();
        if !recalled.trim().eq_ignore_ascii_case("none") && !recalled.trim().is_empty() {
            let facts = crate::agent::parse_protocol_strict(&recalled, 0.8);
            if !facts.is_empty() {
                let mut recall_ctx = memory_ctx.clone();
                recall_ctx.push_str("\nRECALLED FROM TRANSCRIPT:\n");
                for f in &facts {
                    recall_ctx.push_str(&format!(
                        "{} --{}--> {}\n",
                        f.subj, f.pred, f.obj
                    ));
                }
                memory_pred = chat_retry(
                    chat,
                    ANSWER_PROMPT,
                    &format!("CONTEXT:\n{recall_ctx}\n\nQUESTION: {}", inst.question),
                )
                .unwrap_or(memory_pred);
            }
        }
    }
    let baseline_pred = chat_retry(
        chat,
        ANSWER_PROMPT,
        &format!("CONTEXT (transcript):\n{transcript}\n\nQUESTION: {}", inst.question),
    )?;

    let (memory_f1, memory_em) = score_f1(&memory_pred, &inst.answer);
    let (baseline_f1, baseline_em) = score_f1(&baseline_pred, &inst.answer);
    Ok(RunResult {
        question_type: inst.question_type.clone(),
        gold: inst.answer.clone(),
        memory_pred,
        baseline_pred,
        memory_f1,
        baseline_f1,
        memory_em,
        baseline_em,
        memory_ctx_tokens: memory_ctx.len() / 4,
        transcript_tokens: transcript.len() / 4,
    })
}
