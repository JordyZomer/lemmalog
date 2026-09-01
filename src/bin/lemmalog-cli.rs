//! lemmalog-cli: headless memory access on the same snapshot the MCP
//! server uses (`LEMMALOG_MCP_PATH`). For contexts where MCP tools are
//! unreachable — CLI sub-agents that don't inherit MCP connections,
//! scripts, cron jobs — every mutation here is visible to the next MCP
//! load and vice versa.
//!
//!   lemmalog-cli observe  --facts 'alice --works_at--> acme'
//!   lemmalog-cli query   --goal 'current("alice", R, O)'
//!   lemmalog-cli retract --facts 'alice --works_at--> acme'
//!   lemmalog-cli context --query 'where does alice work'
//!   lemmalog-cli why     --fact 'reports_to(alice, carol)'
//!   lemmalog-cli rules   --rules 'reach(X,Z) :- current(X,"dep",Y), reach(Y,Z).'
//!   lemmalog-cli dump    [--pred current]
//!
//! Concurrency: load → mutate → save is atomic-rename, but the CLI and a
//! live MCP server hold separate in-process copies — coordinate so only
//! one writes at a time (sub-agent runs while the parent only reads),
//! or have every writer use the CLI.
//!
//! Env: LEMMALOG_MCP_PATH (snapshot path; required for mutations,
//! optional for fresh-session queries).

#![cfg(feature = "mcp")]

use lemmalog::agent::{AgentMemory, MockExtractor};
use std::io::Read;

fn snap_path() -> String {
    std::env::var("LEMMALOG_MCP_PATH").unwrap_or_else(|_| {
        eprintln!("lemmalog-cli: set LEMMALOG_MCP_PATH to the shared snapshot path");
        std::process::exit(2);
    })
}

fn load(path: &str) -> AgentMemory<MockExtractor> {
    match AgentMemory::load(MockExtractor::new(0.9), path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("lemmalog-cli: snapshot load failed ({e}); starting fresh");
            AgentMemory::new(MockExtractor::new(0.9), "").expect("fresh memory")
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| {
        args.get(i + 1).cloned().filter(|v| !v.starts_with("--"))
    })
}

fn stdin_or_flag(args: &[String], name: &str) -> String {
    if let Some(v) = flag(args, name) {
        return v;
    }
    let mut buf = String::new();
    let _ = std::io::stdin().read_to_string(&mut buf);
    buf
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    match cmd.as_str() {
        "observe" => {
            let mut m = load(&snap_path());
            let text = stdin_or_flag(&args, "--facts");
            let ts = flag(&args, "--ts").and_then(|t| t.parse::<i64>().ok());
            let ts = ts.unwrap_or(m.engine.now);
            let (report, dropped) = m.observe_extracted(&text, ts);
            let _ = m.maintain(m.engine.now);
            m.save(&snap_path()).expect("save snapshot");
            println!(
                "added={} updated={} noop={} escalations={}",
                report.added, report.updated, report.noop, report.escalations.len()
            );
            for d in dropped.iter().take(5) {
                println!("dropped: {} ({})", d.0, d.1);
            }
        }
        "retract" => {
            let mut m = load(&snap_path());
            let text = stdin_or_flag(&args, "--facts");
            let (done, missing, died) = m.retract_facts(&text);
            m.save(&snap_path()).expect("save snapshot");
            println!("retracted {} fact(s)", done.len());
            if !died.is_empty() {
                println!("{} derived fact(s) died:", died.len());
                for d in died.iter().take(15) {
                    println!("  {d}");
                }
            }
            for mi in &missing {
                println!("not found: {mi}");
            }
        }
        "query" => {
            let m = load(&snap_path());
            let goal = stdin_or_flag(&args, "--goal");
            match m.ask(&goal) {
                Ok(rows) => {
                    if rows.is_empty() {
                        println!("(no answers — asserted facts are current(S, rel, O))");
                    } else {
                        println!("{}", rows.join("\n"));
                    }
                }
                Err(e) => {
                    eprintln!("parse: {e}\nhint: quote entity names — bare capitalized words are variables");
                    std::process::exit(1);
                }
            }
        }
        "context" => {
            let m = load(&snap_path());
            let query = stdin_or_flag(&args, "--query");
            let budget = flag(&args, "--budget")
                .and_then(|b| b.parse::<usize>().ok())
                .unwrap_or(1000);
            print!("{}", m.context_for_query_rich(&query, budget));
        }
        "why" => {
            let m = load(&snap_path());
            let fact = stdin_or_flag(&args, "--fact");
            println!("{}", m.why(&fact));
        }
        "rules" => {
            let mut m = load(&snap_path());
            let rules = stdin_or_flag(&args, "--rules");
            match m.install_rules(&rules) {
                Ok(id) => {
                    let n = m.maintain(m.engine.now);
                    m.save(&snap_path()).expect("save snapshot");
                    println!("installed {id}; backfill derived +{n} facts");
                }
                Err(e) => {
                    eprintln!("install: {e}");
                    std::process::exit(1);
                }
            }
        }
        "rmrules" => {
            let mut m = load(&snap_path());
            let id = flag(&args, "--id").expect("--id <batch>");
            if m.uninstall_rules(&id) {
                let _ = m.maintain(m.engine.now);
                m.save(&snap_path()).expect("save snapshot");
                println!("uninstalled {id}; derivations reverted");
            } else {
                eprintln!("no batch {id:?} (see: lemmalog-cli batches)");
                std::process::exit(1);
            }
        }
        "batches" => {
            let m = load(&snap_path());
            for (id, src) in m.rule_batches() {
                println!("{id}: {}", src.lines().next().unwrap_or(""));
            }
        }
        "dump" => {
            let m = load(&snap_path());
            let pred = flag(&args, "--pred");
            let mut preds: Vec<&String> = m.engine.relations.keys().collect();
            preds.sort();
            for p in preds {
                if let Some(want) = &pred {
                    if p != want {
                        continue;
                    }
                }
                for key in m.engine.relation_keys(p) {
                    println!("{}", m.engine.render_fact(p, &key));
                }
            }
        }
        other => {
            eprintln!(
                "usage: lemmalog-cli observe|retract|query|context|why|rules|rmrules|batches|dump [flags]\n\
                 flags: --facts|--goal|--query|--fact|--rules|--id|--pred|--ts|--budget  (or stdin)\n\
                 unknown command {other:?}"
            );
            std::process::exit(2);
        }
    }
}
