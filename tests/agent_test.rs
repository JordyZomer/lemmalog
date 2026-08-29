use lemmalog::{AgentMemory, MockExtractor, Value};

fn mem(extra: &str) -> AgentMemory<MockExtractor> {
    AgentMemory::new(MockExtractor::new(0.9), extra).unwrap()
}

#[test]
fn ingest_adds_facts_with_provenance() {
    let mut m = mem("");
    let r = m.observe("alice --works_at--> acme\nalice --manager--> bob");
    assert_eq!(r.added, 2);
    assert_eq!(m.maintain(100), 2); // 2 current facts
    let rows = m.ask("current(\"alice\", R, O)").unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().any(|r| r.contains("acme")));
    // provenance points at the episode
    let out = m.why("current(alice, works_at, acme)");
    assert!(out.contains("ep1"), "why: {out}");
}

#[test]
fn reobservation_is_noop() {
    let mut m = mem("");
    m.observe("alice --works_at--> acme");
    let r = m.observe("alice --works_at--> acme");
    assert_eq!(r.noop, 1);
    assert_eq!(r.added, 0);
    m.maintain(100);
    assert_eq!(m.ask("current(\"alice\", \"works_at\", O)").unwrap().len(), 1);
}

#[test]
fn exclusive_pred_supersedes_deterministically() {
    let mut m = mem("");
    m.observe("alice --works_at--> acme");
    m.maintain(100);
    assert_eq!(
        m.ask("current(\"alice\", \"works_at\", O)").unwrap(),
        vec!["O=acme".to_string()]
    );
    let r = m.observe_at("alice --works_at--> gigant", 200);
    assert_eq!(r.updated, 1, "deterministic UPDATE, no escalation");
    assert!(r.escalations.is_empty());
    m.maintain(200);
    assert_eq!(
        m.ask("current(\"alice\", \"works_at\", O)").unwrap(),
        vec!["O=gigant".to_string()],
        "knowledge update applied"
    );
    // the old edge is closed, not deleted: history preserved
    let closed = m
        .engine
        .query("edge", &[None, None, None, None, Some(lemmalog::Value::Int(200)), None]);
    assert_eq!(closed.len(), 1, "exactly one edge closed at t=200");
}

#[test]
fn non_exclusive_conflict_escalates() {
    let mut m = mem("");
    m.observe("alice --likes--> bob");
    let r = m.observe("alice --likes--> carol");
    assert_eq!(r.added, 1);
    assert_eq!(r.escalations.len(), 1);
    assert!(r.escalations[0].contains("conflict"));
    assert_eq!(m.escalations().len(), 1);
    m.maintain(100);
    // both remain open pending agent resolution
    assert_eq!(m.ask("current(\"alice\", \"likes\", O)").unwrap().len(), 2);
    m.resolve_escalation(0);
    assert_eq!(m.escalations().len(), 0);
}

#[test]
fn context_assembly_is_positional_and_budgeted() {
    let mut m = mem("");
    m.observe("alice --works_at--> acme\nalice --manager--> bob");
    m.maintain(100);
    let ctx = m.context(&["alice"], 200);
    let dist = ctx.find("== memory").unwrap();
    let src = ctx.find("== source").unwrap();
    assert!(dist < src, "distilled facts before verbatim sources");
    assert!(ctx.contains("alice --works_at--> acme"));
    assert!(ctx.contains("[ep1]"), "verbatim episode text included");

    // tiny budget truncates but keeps both sections
    let tiny = m.context(&["alice"], 15);
    assert!(tiny.contains("== memory") && tiny.contains("== source"));
    assert!(tiny.len() < ctx.len());
}

#[test]
fn derivation_rules_compose_with_ingestion() {
    let mut m = mem(
        "reports_to(X,Y) :- current(X,\"manager\",Y).\n\
         trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
    );
    m.observe("alice --manager--> bob\nbob --manager--> carol");
    let derived = m.maintain(100);
    assert_eq!(derived, 5, "2 current + 2 direct + 1 transitive");
    let rows = m.ask("reports_to(\"alice\", Y)").unwrap();
    assert_eq!(rows.len(), 2); // bob, carol
    let proof = m.why("reports_to(alice, carol)");
    assert!(proof.contains("via trans") && proof.contains("ep1"));
}

#[test]
fn context_reports_new_memory() {
    let mut m = mem("");
    m.observe("alice --works_at--> acme");
    m.maintain(100);
    let ctx = m.context(&["alice"], 200);
    assert!(ctx.contains("new in memory"), "{ctx}");
    assert!(
        ctx.contains("current(alice, works_at, acme)"),
        "derived fact reported: {ctx}"
    );
    // a turn with nothing new has no news section
    m.maintain(200);
    let ctx2 = m.context(&["alice"], 200);
    assert!(!ctx2.contains("new in memory"), "{ctx2}");
    // and a new observation shows up in the next context
    m.observe("alice --manager--> bob");
    m.maintain(300);
    let ctx3 = m.context(&["alice"], 200);
    assert!(ctx3.contains("edge(alice, manager, bob"), "{ctx3}");
}

#[test]
fn llm_extractor_pluggable_and_memoized() {
    use lemmalog::LlmExtractor;
    let mut _calls = 0;
    let mut m = AgentMemory::new(
        LlmExtractor::new(move |prompt| {
            _calls += 1;
            assert!(prompt.contains("Extract the factual triples"));
            Ok("alice --works_at[0.7]--> acme\nbob --manager--> alice".to_string())
        }),
        "",
    )
    .unwrap();
    let r = m.observe_at("any episode text", 100);
    assert_eq!(r.added, 2);
    m.maintain(100);
    // per-fact confidence honored by the protocol
    let emp = m.ask("current(\"alice\", \"works_at\", O)").unwrap();
    assert_eq!(emp, vec!["O=acme".to_string()]);
    let (a, wa, ac) = (m.engine.sym("alice"), m.engine.sym("works_at"), m.engine.sym("acme"));
    let f = m.engine.fact("edge", &[a, wa, ac, Value::Int(100), Value::Int(i64::MAX), Value::Int(100)]).unwrap();
    assert!((f.ann.conf - 0.7).abs() < 1e-9, "conf = {}", f.ann.conf);
    // extraction errors degrade to zero facts, not poison
    // (checked via a second memory below)
    let mut m2 = AgentMemory::new(
        LlmExtractor::new(|_| Err("provider down".to_string())),
        "",
    )
    .unwrap();
    let r2 = m2.observe("whatever");
    assert_eq!(r2.added, 0);
}

#[test]
fn snapshot_roundtrip_rebuilds_derived_relations() {
    let dir = std::env::temp_dir().join("lemmalog-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mem.snapshot");
    let path = path.to_str().unwrap();

    let mut m = mem("reports_to(X,Y) :- current(X,\"manager\",Y).\n\
                    trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).");
    m.observe_at("alice --manager--> bob\nbob --manager--> carol", 100);
    m.observe_at("alice --works_at--> acme", 100);
    m.maintain(100);
    m.observe_at("alice --works_at--> gigant", 200); // supersession
    m.observe_at("alice --likes--> khq\nalice --likes--> zph", 300); // escalation
    let derived = m.maintain(300);
    assert!(derived > 0);
    let before_ctx = m.context(&["alice"], 300);

    m.save(path).unwrap();
    let mut m2 = AgentMemory::load(MockExtractor::new(0.9), path).unwrap();

    // derived relations rebuilt; answers identical
    assert_eq!(
        m2.ask("reports_to(\"alice\", Y)").unwrap(),
        vec!["Y=bob".to_string(), "Y=carol".to_string()]
    );
    assert_eq!(
        m2.ask("current(\"alice\", \"works_at\", O)").unwrap(),
        vec!["O=gigant".to_string()]
    );
    // episodes + escalations survive
    assert_eq!(m2.episodes().len(), 4);
    assert_eq!(m2.escalations().len(), 1);
    // context assembly identical apart from the news section: a freshly
    // loaded memory has nothing "new since load", by design
    let strip_news = |s: &str| -> String {
        match s.find("== memory (distilled") {
            Some(i) => s[i..].to_string(),
            None => s.to_string(),
        }
    };
    assert_eq!(strip_news(&m2.context(&["alice"], 300)), strip_news(&before_ctx));
    // why() still walks to the re-asserted base facts
    let w = m2.why("current(alice, works_at, gigant)");
    assert!(w.contains("asserted (base fact)"), "{w}");
    // and the loaded memory keeps working incrementally
    m2.observe_at("bob --works_at--> initech", 400);
    m2.maintain(400);
    assert_eq!(
        m2.ask("current(\"bob\", \"works_at\", O)").unwrap(),
        vec!["O=initech".to_string()]
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn what_if_lookahead_leaves_store_unchanged() {
    let mut m = mem("reports_to(X,Y) :- current(X,\"manager\",Y).\n\
                    trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).");
    m.observe_at("a --manager--> b", 100);
    m.maintain(100);
    assert_eq!(
        m.ask("reports_to(\"a\", Y)").unwrap(),
        vec!["Y=b".to_string()]
    );

    // what would follow if c managed a?
    let (rows, added) = m
        .what_if("c --manager--> a", "reports_to(\"c\", Y)")
        .unwrap();
    let mut got: Vec<&str> = rows.iter().map(|r| r.strip_prefix("Y=").unwrap()).collect();
    got.sort();
    assert_eq!(got, vec!["a", "b"], "hypothetical closure c->a->b");
    assert!(added >= 2, "assumption would add facts: {added}");

    // store untouched: no c facts, no news pollution
    assert!(m.ask("reports_to(\"c\", Y)").unwrap().is_empty());
    assert!(m.engine.changes_from(m.engine.epoch()).is_empty());
    let ctx = m.context(&["a"], 200);
    assert!(!ctx.contains("reports_to(c"), "{ctx}");
    // and the same query repeated is stable
    let (rows2, _) = m
        .what_if("c --manager--> a", "reports_to(\"c\", Y)")
        .unwrap();
    assert_eq!(rows2.len(), 2);

    // committing the real episode makes it true
    m.observe_at("c --manager--> a", 200);
    m.maintain(200);
    assert_eq!(m.ask("reports_to(\"c\", Y)").unwrap().len(), 2);
}

#[test]
fn parse_protocol_reported_gives_drop_reasons() {
    use lemmalog::agent::parse_protocol_reported;
    let (facts, dropped) = parse_protocol_reported(
        "alice --works_at[0.8]--> acme\n\
         speaker --works_at--> acme\n\
         - acme --has_products--> products? maybe not because generic\n\
         this line has no arrow structure",
        0.9,
    );
    assert_eq!(facts.len(), 1);
    assert_eq!(dropped.len(), 3);
    let reasons: Vec<&str> = dropped.iter().map(|(_, r)| r.as_str()).collect();
    assert!(
        reasons.iter().any(|r| r.contains("role word")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("prose") || r.contains("punctuation")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.contains("no `--rel-->`")),
        "{reasons:?}"
    );
}

#[test]
fn observe_extracted_applies_policy_and_reports_drops() {
    let mut m = mem("");
    let (report, dropped) = m.observe_extracted(
        "alice --works_at--> acme\nspeaker --works_at--> acme",
        100,
    );
    assert_eq!(report.added, 1, "only the clean fact asserted");
    assert_eq!(dropped.len(), 1);
    assert!(dropped[0].1.contains("role word"));
    m.maintain(100);
    assert_eq!(
        m.ask("current(\"alice\", \"works_at\", O)").unwrap(),
        vec!["O=acme".to_string()]
    );
}

#[test]
fn source_references_are_valid_entities() {
    use lemmalog::agent::parse_protocol_reported;
    let (facts, dropped) = parse_protocol_reported(
        "sketch_mode --located--> src/features/sketchMode/bind.ts\n\
         entity_token_problem --defined_at--> src/agent.rs:92\n\
         parser --tracked_by--> JordyZomer/lemmalog#12\n\
         sketch_mode --depends_on--> engine_scene",
        1.0,
    );
    assert_eq!(facts.len(), 4, "dropped: {dropped:?}");
    assert!(dropped.is_empty(), "{dropped:?}");
    assert_eq!(facts[1].obj, "src/agent.rs:92");
}

#[test]
fn punctuation_with_spaces_is_still_prose() {
    use lemmalog::agent::parse_protocol_reported;
    let (facts, dropped) = parse_protocol_reported(
        "thing --located--> see src/agent.rs, around line 118\n\
         thing --noted--> Yes. Confirmed.",
        1.0,
    );
    assert!(facts.is_empty(), "{facts:?}");
    assert_eq!(dropped.len(), 2);
    assert!(
        dropped.iter().all(|(_, r)| r.contains("prose")),
        "{dropped:?}"
    );
}
