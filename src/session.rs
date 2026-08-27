//! Interactive session layer: a line-oriented command surface over the
//! engine, shared by the `lemmalog` REPL binary and tests.
//!
//! Commands:
//! - `rule <clause>`    install one rule or fact clause (rule program)
//! - `+ <fact> [@conf] [#prov]`  assert a base fact, e.g.
//!                               `+ edge(alice, works_at, acme, 0, MAX, 1) @0.9 #ep1`
//! - `? <goal>`         query materialized relations (index-backed `ask`)
//! - `?? <goal>`        demand-driven query (magic sets)
//! - `why <fact>`       provenance proof tree
//! - `run`              run one maintenance epoch
//! - `now <t>`          set the clock used by `now(T)` rules
//! - `dump [pred]`      list facts with confidence and provenance
//! - `batches` / `rm <id>`  rule-batch registry (install via `rule` batches)
//! - `esc`              list escalations (agent memory sessions)
//! - `help`             command summary
//!
//! `MAX` in a fact abbreviates the i64 sentinel for open intervals.

use crate::ast::parse_program;
use crate::eval::{Ann, Engine};
use crate::intern::Value;
use std::fmt::Write as _;

pub struct Session {
    pub engine: Engine,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Session {
            engine: Engine::new(),
        }
    }

    /// Execute one command line; returns the output (or an error message).
    pub fn execute(&mut self, line: &str) -> String {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return String::new();
        }
        let (cmd, rest) = match line.split_once(' ') {
            Some((c, r)) => (c, r.trim()),
            None => (line, ""),
        };
        match cmd {
            "help" => Ok(Self::help()),
            "rule" => self.rule(rest),
            "+" => self.assert_fact(rest),
            "?" => self.query(rest, false),
            "??" => self.query(rest, true),
            "why" => self.why(rest),
            "run" => Ok(format!("+{} facts (epoch {})\n", self.engine.run(), self.engine.epoch())),
            "now" => match rest.parse::<i64>() {
                Ok(t) => {
                    self.engine.set_now(t);
                    Ok(format!("now = {t}\n"))
                }
                Err(_) => Err(format!("now: expected an integer, got {rest:?}")),
            },
            "dump" => Ok(self.dump(rest)),
            "batches" => Ok(self
                .engine
                .batches()
                .into_iter()
                .map(|(id, src)| format!("{id}: {src}"))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"),
            "rm" => {
                if self.engine.uninstall(rest) {
                    Ok(format!("uninstalled {rest} (run to recompute)\n"))
                } else {
                    Err(format!("rm: no batch {rest:?}"))
                }
            }
            "esc" => Ok("(escalations live on AgentMemory, not the raw engine)\n".to_string()),
            other => Err(format!("unknown command {other:?} — try `help`")),
        }
        .unwrap_or_else(|e| format!("error: {e}\n"))
    }

    fn help() -> String {
        "rule <clause>            install a rule/fact clause (batched)\n\
         + <fact> [@conf] [#prov] assert a base fact (MAX = open interval)\n\
         ? <goal>                 query materialized relations\n\
         ?? <goal>                demand-driven query (magic sets)\n\
         why <fact>               provenance proof tree\n\
         run                      run one maintenance epoch\n\
         now <t>                  set the rule clock\n\
         dump [pred]              list facts\n\
         batches / rm <id>        rule batch registry\n"
            .to_string()
    }

    fn rule(&mut self, src: &str) -> Result<String, String> {
        let mut text = src.trim().to_string();
        if !text.ends_with('.') {
            text.push('.');
        }
        let id = self
            .engine
            .install_program(&text)
            .map_err(|e| format!("rule: {e}"))?;
        Ok(format!("installed batch {id}\n"))
    }

    /// `+ edge(a, works_at, b, 0, MAX, 100) @0.9 #ep1`
    fn assert_fact(&mut self, rest: &str) -> Result<String, String> {
        let mut conf = 1.0f64;
        let mut prov: Vec<String> = Vec::new();
        let mut core = String::new();
        for tok in rest.split_whitespace() {
            if let Some(c) = tok.strip_prefix('@') {
                conf = c.parse().map_err(|_| format!("+ : bad confidence {c:?}"))?;
            } else if let Some(p) = tok.strip_prefix('#') {
                prov.push(p.to_string());
            } else {
                core.push_str(tok);
                core.push(' ');
            }
        }
        let mut text = core.trim().to_string();
        if !text.ends_with('.') {
            text.push('.');
        }
        let text = text.replace("MAX", &i64::MAX.to_string());
        let clauses = parse_program(&text).map_err(|e| format!("+ : {e}"))?;
        if clauses.len() != 1 || !clauses[0].is_fact {
            return Err(format!("+ : expected one ground fact, got {rest:?}"));
        }
        let head = &clauses[0].head;
        // symbols resolve against the engine's interner by interning now
        let mut fixed = Vec::with_capacity(head.args.len());
        for t in &head.args {
            match t {
                crate::intern::Term::Sym(s) => fixed.push(self.engine.sym(s)),
                crate::intern::Term::Int(i) => fixed.push(Value::Int(*i)),
                _ => return Err(format!("+ : fact must be ground: {rest:?}")),
            }
        }
        let ann = if prov.is_empty() && conf == 1.0 {
            Ann::unit()
        } else {
            Ann::base(conf, prov)
        };
        self.engine.declare(&head.pred, &fixed, ann);
        Ok(format!("{} {}\n", head.pred, fixed.len()))
    }

    fn query(&mut self, goal: &str, deep: bool) -> Result<String, String> {
        // ask()/ask_deep() append the clause terminator themselves
        let text = goal.trim().to_string();
        let rows = if deep {
            self.engine
                .ask_deep(&text)
                .map_err(|e| format!("?? : {e}"))?
        } else {
            self.engine.ask(&text).map_err(|e| format!("? : {e}"))?
        };
        if rows.is_empty() {
            return Ok("(no answers)\n".to_string());
        }
        let mut out = String::new();
        for r in rows {
            let _ = writeln!(out, "{r}");
        }
        Ok(out)
    }

    fn why(&mut self, fact: &str) -> Result<String, String> {
        let mut text = fact.trim().to_string();
        if !text.ends_with('.') {
            text.push('.');
        }
        let clauses = parse_program(&text).map_err(|e| format!("why : {e}"))?;
        if clauses.len() != 1 {
            return Err(format!("why : expected one fact, got {fact:?}"));
        }
        let head = &clauses[0].head;
        let mut args = Vec::with_capacity(head.args.len());
        for t in &head.args {
            match t {
                crate::intern::Term::Sym(s) => args.push(self.engine.sym(s)),
                crate::intern::Term::Int(i) => args.push(Value::Int(*i)),
                _ => return Err(format!("why : fact must be ground: {fact:?}")),
            }
        }
        Ok(self.engine.why(&head.pred, &args))
    }

    fn dump(&self, pred: &str) -> String {
        let mut preds: Vec<&String> = self.engine.relations.keys().collect();
        preds.sort();
        let mut out = String::new();
        for p in preds {
            if !pred.is_empty() && p != pred {
                continue;
            }
            if let Some(rel) = self.engine.relations.get(p) {
                for row in &rel.rows {
                    let prov: Vec<&String> = row.fact.ann.prov.iter().collect();
                    let _ = writeln!(
                        out,
                        "{} (conf {:.2}, prov {prov:?})",
                        self.engine.render_fact(p, &row.key),
                        row.fact.ann.conf
                    );
                }
            }
        }
        if out.is_empty() {
            out.push_str("(empty)\n");
        }
        out
    }
}
