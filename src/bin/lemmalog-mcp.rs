//! lemmalog-mcp: the Lemmalog engine as an MCP server (stdio JSON-RPC),
//! for agent CLIs like Claude Code and Kimi CLI.
//!
//! Build:  cargo build --release --features mcp
//! Register (Claude Code):  claude mcp add lemmalog -- <path>/lemmalog-mcp
//! Register (Kimi CLI):     kimi mcp add lemmalog -- <path>/lemmalog-mcp
//!
//! Persistence: set LEMMALOG_MCP_PATH=/tmp/lemmalog.snapshot to keep
//! memory across server restarts (saved after every mutating call).
//!
//! The host model does the extraction (it reads the conversation anyway)
//! and asserts triples via `observe`; Lemmalog derives closures,
//! temporal views, canonicalizations, aggregations and answers `query`
//! and `why` deterministically.
//!
//! Time is bitemporal and the two clocks are kept apart: an `observe`
//! carries the *valid-from* time of the facts it asserts, while every read
//! path syncs the engine clock to the wall clock before deriving (see
//! `sync_clock`). Reads therefore answer as of the present no matter what
//! order episodes were ingested in, and backdating a batch cannot hide
//! facts asserted after it.

#![cfg(feature = "mcp")]

use lemmalog::agent::AgentMemory;
use lemmalog::canonical;
use lemmalog::eval::Engine;
use lemmalog::intern::Value;
use serde_json::{json, Value as J};
use std::io::{BufRead, Write};

struct State {
    memory: AgentMemory<lemmalog::agent::MockExtractor>,
}

fn main() {
    let path = std::env::var("LEMMALOG_MCP_PATH").ok();
    let mut memory = AgentMemory::new(
        lemmalog::agent::MockExtractor::new(0.9),
        "",
    )
    .expect("fresh memory");
    if let Some(p) = &path {
        if std::path::Path::new(p).exists() {
            match AgentMemory::load(lemmalog::agent::MockExtractor::new(0.9), p) {
                Ok(m) => memory = m,
                Err(e) => eprintln!("lemmalog-mcp: snapshot load failed: {e}"),
            }
        }
    }
    let mut state = State { memory };
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<J>(&line) else { continue };
        let method = msg["method"].as_str().unwrap_or_default().to_string();
        let id = msg.get("id").cloned();
        if id.is_none() {
            continue; // notification (e.g. initialized): no response
        }
        let result = match method.as_str() {
            "initialize" => Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "lemmalog", "version": env!("CARGO_PKG_VERSION")},
                "instructions": SERVER_INSTRUCTIONS
            })),
            "tools/list" => Ok(json!({"tools": tools()})),
            "tools/call" => {
                let name = msg["params"]["name"].as_str().unwrap_or_default();
                let args = &msg["params"]["arguments"];
                tool_call(&mut state, name, args, path.as_deref())
            }
            other => Err(format!("unknown method {other:?}")),
        };
        let resp = match result {
            Ok(v) => json!({"jsonrpc": "2.0", "id": id.unwrap(), "result": v}),
            Err(e) => json!({
                "jsonrpc": "2.0", "id": id.unwrap(),
                "error": {"code": -32000, "message": e}
            }),
        };
        writeln!(out, "{resp}").ok();
        out.flush().ok();
    }
}

fn tools() -> J {
    json!([
        tool("lemmalog_observe",
            "Assert facts into memory (host model does extraction). Input: facts in the line protocol 'S --rel[conf]--> O', one per line; optional ts integer = the facts' valid-from time (default: now). Backdating with ts is safe: it sets valid-time only, never the clock reads use. Example: 'Alice --works_at--> Acme\\nBob --manager--> Carol'.", &["facts", "ts"], &["facts"]),
        tool("lemmalog_retract",
            "Retract facts that turned out to be WRONG (line protocol, same as observe). Open matching rows are removed and invalidation propagates: the response reports which derived facts died as a consequence. For a value that merely CHANGED, prefer re-asserting the same relation (the update policy supersedes).", &["facts"], &["facts"]),
        tool("lemmalog_query",
            "Query derived memory: a goal atom like 'reports_to(\"Alice\", Y)' or 'current(X, works_at, O)'. Returns variable bindings. Read-only.", &["goal"], &["goal"]),
        tool("lemmalog_query_deep",
            "Demand-driven query (magic sets) for exploratory points: same goal syntax as lemmalog_query.", &["goal"], &["goal"]),
        tool("lemmalog_why",
            "Provenance proof tree for a ground fact like 'reports_to(Alice, Carol)' — which rules and source episodes produced it.", &["fact"], &["fact"]),
        tool("lemmalog_install_rules",
            "Install a Datalog rule batch (versioned, revertable). Input: rules text. Rules: 'head(X,Y) :- atom(X,Y), cmp.' with stratified negation (!atom), count/min/max/sum aggregates in heads, now(T) builtin.", &["rules"], &["rules"]),
        tool("lemmalog_uninstall",
            "Uninstall a rule batch by id (see lemmalog_batches). Derivations revert.", &["id"], &["id"]),
        tool("lemmalog_batches", "List installed rule batches.", &[], &[]),
        tool("lemmalog_what_if",
            "Hypothetical: 'what would follow if these facts were true?' Input: facts (line protocol) + goal atom. Store is untouched.", &["facts", "goal"], &["facts", "goal"]),
        tool("lemmalog_canonicalize",
            "Entity resolution: asserts alias edges (line protocol 'local --alias_of[conf]--> canonical'), installs the canonicalization rules and canonical views over current. Conflicts surface as alias_conflict facts.", &["facts"], &["facts"]),
        tool("lemmalog_context",
            "Query-driven context assembly via hybrid retrieval: BM25 over facts and episodes + entity-match boosting, budget-aware. Input: query (natural language) + optional budget_tokens (default 1000). Returns relevance-selected facts and their verbatim source episodes — use this instead of lemmalog_dump when preparing a grounded answer.", &["query", "budget_tokens"], &["query"]),
        tool("lemmalog_dump", "List facts of a predicate (or all) with confidence and provenance.", &["pred"], &[]),
        tool("lemmalog_changes",
            "Resync after a context reset or another agent's work: everything asserted, derived, or retracted since an epoch. Input: optional `since` epoch integer (default: 0 = everything, capped). The response carries the current epoch — checkpoint it and pass it back next time.", &["since"], &[]),
        tool("lemmalog_save", "Persist memory to LEMMALOG_MCP_PATH.", &[], &[]),
        tool("lemmalog_run", "Sync the clock to the present and run one maintenance epoch (usually automatic — every read does it).", &[], &[]),
    ])
}

/// Sent on initialize; clients prepend it to the model's context. Small
/// models never read the README — they read this.
const SERVER_INSTRUCTIONS: &str = "\
Lemmalog is your working memory: assert facts as you verify them \
(S --rel[conf]--> O), derive with rules, and when a fact turns out \
WRONG call lemmalog_retract (the response lists which conclusions \
died). Asserted facts are queried as current(Subject, \"rel\", Object) \
— querying rel(X, Y) directly returns nothing. For a changed value, \
re-assert the same relation (old one supersedes). lemmalog_changes \
with your last epoch resyncs you after context resets.";

/// Property descriptions shared by the per-tool schemas (each tool
/// advertises only the properties it actually takes — a model that sees
/// `query` on lemmalog_observe will eventually pass it one).
fn prop_desc(p: &str) -> &'static str {
    match p {
        "facts" => "line-protocol facts",
        "goal" => "query atom",
        "fact" => "ground fact",
        "rules" => "rule program text",
        "id" => "batch id",
        "pred" => "predicate name",
        "ts" => "timestamp",
        "query" => "natural-language query",
        "budget_tokens" => "context token budget",
        "since" => "epoch checkpoint",
        _ => "",
    }
}

fn tool(name: &str, desc: &str, props: &[&str], required: &[&str]) -> J {
    let properties: serde_json::Map<String, J> = props
        .iter()
        .map(|p| {
            (
                p.to_string(),
                json!({"type": if *p == "ts" || *p == "budget_tokens" || *p == "since" { "integer" } else { "string" }, "description": prop_desc(p)}),
            )
        })
        .collect();
    json!({
        "name": name,
        "description": desc,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    })
}

/// Parse `pred(Arg1, Arg2, ...)` into a predicate and bare argument
/// strings (quotes stripped). Every argument must be a constant — entity
/// names arrive ground regardless of case.
fn parse_fact_atom(s: &str) -> Result<(String, Vec<String>), String> {
    let s = s.trim().trim_end_matches('.');
    let open = s.find('(').ok_or_else(|| format!("expected pred(args): {s:?}"))?;
    let close = s.rfind(')').ok_or_else(|| format!("expected pred(args): {s:?}"))?;
    if close <= open {
        return Err(format!("expected pred(args): {s:?}"));
    }
    let pred = s[..open].trim().to_string();
    if pred.is_empty() {
        return Err(format!("missing predicate: {s:?}"));
    }
    let args: Vec<String> = s[open + 1..close]
        .split(',')
        .map(|a| a.trim().trim_matches('"').to_string())
        .map(|a| a.trim_matches('"').to_string())
        .collect();
    if args.iter().any(|a| a.is_empty()) {
        return Err(format!("empty argument in {s:?} (why needs a ground fact)"));
    }
    Ok((pred, args))
}

fn engine_of(state: &mut State) -> &mut Engine {
    &mut state.memory.engine
}

/// Wall-clock seconds since the Unix epoch.
fn wall_clock() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Advance the engine clock to the present and re-derive temporal views.
///
/// `now` is the *reader's* present, not a side effect of the last write.
/// Without this, `engine.now` is whatever timestamp the last `observe`
/// passed, so a backdated ingest silently hides every fact asserted at a
/// later `ts` from `current/3` — the fact stays in `edge`, but fails the
/// `VF =< T` guard. Reads sync forward; ingest keeps honouring its own
/// `ts` as valid-time. The clock only ever moves forward here: a
/// future-dated fact must not drag the present backwards.
fn sync_clock(state: &mut State) {
    let t = wall_clock();
    let e = &mut state.memory.engine;
    if t > e.now {
        e.set_now(t);
        e.invalidate_derived();
        let _ = e.run();
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n).collect();
        format!("{cut}...")
    }
}

/// Empty query results must redirect, not dead-end: the classic trap is
/// querying the raw relation (`depends_on(X, Y)`) when asserted facts
/// live under `current(S, rel, O)` — the answer is silently empty. Turn
/// that into the hint that fixes the next call.
fn empty_query_hint(e: &Engine, goal: &str) -> String {
    let pred = goal.split('(').next().unwrap_or_default().trim();
    if e.relations.contains_key(pred) {
        return "(no answers — the relation exists but holds no rows; \
check spelling of constant arguments, or run lemmalog_dump on it)"
            .to_string();
    }
    // is `pred` used as a RELATION inside current? then the fact shape
    // is the trap
    for key in e.relation_keys("current") {
        if key.len() == 3 && e.interner.display(&key[1]) == pred {
            return format!(
                "(no answers — `{pred}` is a RELATION, not a predicate: asserted \
facts are current(S, rel, O). Try current(X, \"{pred}\", Y)"
            );
        }
    }
    // unknown predicate: closest names by shared stem
    let stem = |s: &str| s.split('_').next().unwrap_or(s).to_lowercase();
    let similar: Vec<String> = e
        .relations
        .keys()
        .filter(|r| !r.starts_with("__") && stem(r) == stem(pred))
        .cloned()
        .take(5)
        .collect();
    if similar.is_empty() {
        format!(
            "(no answers — no relation named `{pred}`; asserted facts live \
under current(S, rel, O), derived under their rule heads)"
        )
    } else {
        format!(
            "(no answers — no relation named `{pred}`; similar: {}. \
Asserted facts live under current(S, rel, O)",
            similar.join(", ")
        )
    }
}

fn tool_call(
    state: &mut State,
    name: &str,
    args: &J,
    path: Option<&str>,
) -> Result<J, String> {
    // Input errors (bad goal syntax, rejected rules, unknown batch id)
    // return as tool results with isError: true so the model can
    // self-correct; JSON-RPC errors are reserved for unknown tools and
    // server faults.
    let inner: Result<String, String> = match name {
        "lemmalog_observe" => {
            let facts = args["facts"].as_str().unwrap_or_default();
            // No ts means "now" — the wall clock, never the engine clock,
            // which is only ever a record of the last write.
            let ts = args["ts"].as_i64().unwrap_or_else(wall_clock);
            let mem = &mut state.memory;
            let (report, dropped) = mem.observe_extracted(facts, ts);
            // Facts keep `ts` as their valid-time (VF), but the derived view
            // is as of the present: backdating an episode must not rewind
            // `current/3` past everything already asserted.
            let now = wall_clock().max(mem.engine.now);
            if now > mem.engine.now {
                mem.engine.invalidate_derived();
            }
            mem.maintain(now);
            let mut out = format!(
                "added={} updated={} noop={} escalations={}",
                report.added, report.updated, report.noop, report.escalations.len()
            );
            for e in report.escalations.iter().take(3) {
                out.push_str(&format!("\nescalation: {e}"));
            }
            if report.escalations.len() > 3 {
                out.push_str(&format!(
                    "\n(+{} more escalations)",
                    report.escalations.len() - 3
                ));
            }
            if !dropped.is_empty() {
                out.push_str(&format!(
                    "\ndropped {} line(s) — NOT asserted:",
                    dropped.len()
                ));
                for (line, reason) in dropped.iter().take(5) {
                    out.push_str(&format!(
                        "\n  `{}` — {}",
                        truncate(line, 80),
                        reason
                    ));
                }
                if dropped.len() > 5 {
                    out.push_str(&format!("\n  (+{} more)", dropped.len() - 5));
                }
            }
            Ok(out)
        }
        "lemmalog_retract" => {
            let facts = args["facts"].as_str().unwrap_or_default();
            if facts.trim().is_empty() {
                Err("input: `facts` is required — line-protocol facts to retract".to_string())
            } else {
                // invalidation is computed against the derived view, so the
                // clock has to be current before we decide what dies
                sync_clock(state);
                let (done, missing, died) = state.memory.retract_facts(facts);
                let mut out = String::new();
                if !done.is_empty() {
                    out.push_str(&format!(
                        "retracted {} fact(s); invalidation propagated:\n",
                        done.len()
                    ));
                    if died.is_empty() {
                        out.push_str("  no derived facts depended on them\n");
                    } else {
                        out.push_str(&format!(
                            "  {} derived fact(s) died:\n",
                            died.len()
                        ));
                        for d in died.iter().take(15) {
                            out.push_str(&format!("    {d}\n"));
                        }
                        if died.len() > 15 {
                            out.push_str(&format!("    (+{} more)\n", died.len() - 15));
                        }
                    }
                }
                if !missing.is_empty() {
                    out.push_str(&format!(
                        "not found (no open fact matches):\n  {}",
                        missing.join("\n  ")
                    ));
                }
                if done.is_empty() && missing.is_empty() {
                    out.push_str(
                        "nothing retracted — every line failed to parse \
(strict validation: S --rel--> O with real entity names)",
                    );
                }
                Ok(out)
            }
        }
        "lemmalog_changes" => {
            let since = args["since"].as_i64().unwrap_or(0).max(0) as u64;
            sync_clock(state);
            let e = engine_of(state);
            let events = e.changes_since(since);
            let epoch = e.epoch();
            let mut out = format!("epoch={epoch}\n");
            if events.is_empty() {
                out.push_str("no changes since that epoch");
            } else {
                let shown = events.len().min(200);
                for ev in events.iter().take(shown) {
                    match ev {
                        lemmalog::eval::Change::Added(_, (p, k)) => {
                            out.push_str(&format!("+ {}", e.render_fact(p, k)))
                        }
                        lemmalog::eval::Change::Retracted(_, (p, k)) => {
                            out.push_str(&format!("- {}", e.render_fact(p, k)))
                        }
                        lemmalog::eval::Change::Cleared(_, p) => {
                            out.push_str(&format!("- (cleared {p})"))
                        }
                    };
                    out.push('\n');
                }
                if events.len() > shown {
                    out.push_str(&format!("(+{} more)\n", events.len() - shown));
                }
                out.push_str(&format!(
                    "checkpoint: pass since={epoch} next time"
                ));
            }
            Ok(out)
        }
        "lemmalog_query" => {
            let goal = args["goal"].as_str().unwrap_or_default().to_string();
            sync_clock(state);
            match engine_of(state).ask(&goal) {
                Ok(rows) => Ok(if rows.is_empty() {
                    empty_query_hint(engine_of(state), &goal)
                } else {
                    rows.join("\n")
                }),
                Err(e) => Err(format!(
                    "parse: could not parse goal `{}`\nreason: {e}\nhint: quote entity names — bare capitalized words are variables. Example: reports_to(\"Alice\", Y)",
                    truncate(&goal, 120)
                )),
            }
        }
        "lemmalog_query_deep" => {
            let goal = args["goal"].as_str().unwrap_or_default().to_string();
            sync_clock(state);
            match engine_of(state).ask_deep(&goal) {
                Ok(rows) => Ok(if rows.is_empty() {
                    empty_query_hint(engine_of(state), &goal)
                } else {
                    rows.join("\n")
                }),
                Err(e) => Err(format!(
                    "query: could not evaluate `{}`\nreason: {e}\nhint: quote entity names — bare capitalized words are variables. Example: reports_to(\"Alice\", Y)",
                    truncate(&goal, 120)
                )),
            }
        }
        "lemmalog_why" => {
            let fact = args["fact"].as_str().unwrap_or_default().to_string();
            sync_clock(state);
            match parse_fact_atom(&fact) {
                Ok((pred, parts)) => {
                    let vals: Vec<Value> = parts
                        .iter()
                        .map(|a| match a.parse::<i64>() {
                            Ok(i) => Value::Int(i),
                            Err(_) => state.memory.engine.sym(a),
                        })
                        .collect();
                    Ok(state.memory.engine.why(&pred, &vals))
                }
                Err(e) => Err(format!(
                    "input: {e}\nexample: reports_to(Alice, Carol)"
                )),
            }
        }
        "lemmalog_install_rules" => {
            let rules = args["rules"].as_str().unwrap_or_default();
            sync_clock(state);
            match engine_of(state).install_program(rules) {
                Ok(id) => {
                    let n = engine_of(state).run();
                    Ok(format!("installed {id}; backfill derived +{n} facts"))
                }
                Err(e) => Err(format!(
                    "rules rejected — nothing installed:\n{e}\ncommon causes: recursion through negation; aggregates outside rule heads; parse errors (rules end with '.')"
                )),
            }
        }
        "lemmalog_uninstall" => {
            let id = args["id"].as_str().unwrap_or_default().to_string();
            sync_clock(state);
            if engine_of(state).uninstall(&id) {
                let _ = engine_of(state).run();
                Ok(format!("uninstalled {id}; derivations recomputed"))
            } else {
                Err(format!(
                    "input: no batch {id:?} — call lemmalog_batches to list installed ids"
                ))
            }
        }
        "lemmalog_batches" => Ok(engine_of(state)
            .batches()
            .into_iter()
            .map(|(id, src)| format!("{id}: {src}"))
            .collect::<Vec<_>>()
            .join("\n")),
        "lemmalog_what_if" => {
            let facts = args["facts"].as_str().unwrap_or_default();
            let goal = args["goal"].as_str().unwrap_or_default().to_string();
            let (cands, dropped) = lemmalog::agent::parse_protocol_reported(facts, 0.9);
            if cands.is_empty() && !facts.trim().is_empty() {
                let reasons: Vec<String> = dropped
                    .iter()
                    .map(|(l, r)| format!("  `{}` — {}", truncate(l, 60), r))
                    .collect();
                return Ok(json!({
                    "content": [{"type": "text", "text": format!(
                        "input: no valid facts in the hypothetical.\n{}",
                        reasons.join("\n")
                    )}],
                    "isError": true
                }));
            }
            sync_clock(state);
            let e = engine_of(state);
            let now = e.now;
            let mut extra: Vec<(String, Vec<Value>)> = Vec::new();
            for c in &cands {
                let (s, p, o) = (e.sym(&c.subj), e.sym(&c.pred), e.sym(&c.obj));
                extra.push((
                    "edge".to_string(),
                    vec![
                        s, p, o,
                        Value::Int(now),
                        Value::Int(i64::MAX),
                        Value::Int(now),
                    ],
                ));
            }
            let refs: Vec<(&str, &[Value])> = extra
                .iter()
                .map(|(p, a)| (p.as_str(), a.as_slice()))
                .collect();
            match e.hypothetical(&refs, &goal) {
                Ok(rows) => Ok(if rows.is_empty() {
                    "(no answers)".to_string()
                } else {
                    rows.join("\n")
                }),
                Err(er) => Err(format!(
                    "query: could not evaluate `{}`\nreason: {er}\nhint: quote entity names — bare capitalized words are variables",
                    truncate(&goal, 120)
                )),
            }
        }
        "lemmalog_canonicalize" => {
            let facts = args["facts"].as_str().unwrap_or_default();
            let aliases: Vec<_> = lemmalog::agent::parse_protocol(facts, 0.9)
                .into_iter()
                .filter(|c| c.pred == "alias_of")
                .collect();
            if aliases.is_empty() {
                return Ok(json!({
                    "content": [{"type": "text", "text": "input: no alias edges found — lines must look like `local --alias_of[0.9]--> canonical`"}],
                    "isError": true
                }));
            }
            sync_clock(state);
            let e = engine_of(state);
            for c in &aliases {
                canonical::assert_alias(e, &c.subj, &c.obj, c.confidence);
            }
            match canonical::install_canonicalization(e, &["current"]) {
                Ok(_) => {
                    let now = state.memory.engine.now;
                    let _ = state.memory.maintain(now);
                    let conflicts = canonical::alias_conflicts(&state.memory.engine);
                    if conflicts.is_empty() {
                        Ok("canonicalization installed; no conflicts".to_string())
                    } else {
                        Ok(format!(
                            "canonicalization installed; CONFLICTS (retract bad alias edges):\n{}",
                            conflicts.join("\n")
                        ))
                    }
                }
                Err(er) => Err(format!(
                    "canonicalization rejected: {er}"
                )),
            }
        }
        "lemmalog_context" => {
            let query = args["query"].as_str().unwrap_or_default().to_string();
            let budget = args["budget_tokens"].as_i64().unwrap_or(1000).max(50) as usize;
            if query.trim().is_empty() {
                Err("input: `query` is required — the natural-language question the context should serve".to_string())
            } else {
                sync_clock(state);
                Ok(state.memory.context_for_query_rich(&query, budget))
            }
        }
        "lemmalog_dump" => {
            let pred = args["pred"].as_str().unwrap_or_default();
            sync_clock(state);
            let e = engine_of(state);
            let mut preds: Vec<&String> = e.relations.keys().collect();
            preds.sort();
            let mut out = String::new();
            for p in preds {
                if !pred.is_empty() && p != pred {
                    continue;
                }
                for key in e.relation_keys(p) {
                    out.push_str(&format!("{}\n", e.render_fact(p, &key)));
                }
            }
            if out.is_empty() {
                out.push_str("(empty)\n");
            }
            Ok(out)
        }
        "lemmalog_save" => {
            let p = path.ok_or_else(|| {
                "input: LEMMALOG_MCP_PATH not set — register the server with --env LEMMALOG_MCP_PATH=...".to_string()
            })?;
            match state.memory.save(p) {
                Ok(_) => Ok(format!("saved to {p}")),
                Err(e) => Err(format!("io: snapshot save failed: {e}")),
            }
        }
        "lemmalog_run" => {
            sync_clock(state);
            let n = engine_of(state).run();
            Ok(format!("+{n} facts (clock {})", engine_of(state).now))
        }
        other => {
            // models invent tool names; teach instead of rejecting —
            // suggest the closest real tool so the next call succeeds
            let known: Vec<&str> = [
                "lemmalog_observe", "lemmalog_retract", "lemmalog_query",
                "lemmalog_query_deep", "lemmalog_why", "lemmalog_install_rules",
                "lemmalog_uninstall", "lemmalog_batches", "lemmalog_what_if",
                "lemmalog_canonicalize", "lemmalog_context", "lemmalog_dump",
                "lemmalog_changes", "lemmalog_save", "lemmalog_run",
            ]
            .to_vec();
            let stripped = other.trim_start_matches("lemmalog_");
            let close: Vec<&str> = known
                .iter()
                .filter(|t| {
                    let b = t.trim_start_matches("lemmalog_");
                    b.contains(stripped) || stripped.contains(b)
                })
                .copied()
                .collect();
            let hint = if close.is_empty() {
                format!("unknown tool {other:?}; available: {}", known.join(", "))
            } else {
                format!(
                    "unknown tool {other:?}; did you mean {}? available: {}",
                    close.join(" or "),
                    known.join(", ")
                )
            };
            return Ok(json!({
                "content": [{"type": "text", "text": hint}],
                "isError": true
            }));
        }
    };
    let (text, is_error) = match inner {
        Ok(t) => (t, false),
        Err(t) => (t, true),
    };
    if !is_error
        && matches!(
            name,
            "lemmalog_observe" | "lemmalog_retract" | "lemmalog_install_rules"
                | "lemmalog_uninstall" | "lemmalog_canonicalize"
        )
    {
        if let Some(p) = path {
            let _ = state.memory.save(p);
        }
    }
    Ok(json!({
        "content": [{"type": "text", "text": text}],
        "isError": is_error
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Count your tokens before you count your tools": the wire surface
    /// every model pays for on every turn. Guard against drift into the
    /// small-model failure mode (a 126-tool surface costs ~40K tokens;
    /// ours should stay a small fraction of that — if this test fails,
    /// split the server or move to meta-tools, don't just bump it).
    #[test]
    fn tool_surface_stays_under_budget() {
        let wire = serde_json::to_string(&tools()).unwrap();
        let est_tokens = wire.len() / 4; // chars-per-token lower bound
        assert!(
            est_tokens < 4500,
            "tool surface ~{est_tokens} tokens ({} tools, {} bytes) — trim descriptions or consolidate",
            tools().as_array().unwrap().len(),
            wire.len()
        );
    }

    #[test]
    fn every_tool_schema_lists_only_its_own_properties() {
        let ts = tools();
        let list = ts.as_array().unwrap();
        assert_eq!(list.len(), 15);
        for t in list {
            let props: Vec<&str> = t["inputSchema"]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .map(|k| k.as_str())
                .collect();
            assert!(!props.is_empty() || t["name"].as_str().unwrap() != "lemmalog_observe");
        }
    }
}
