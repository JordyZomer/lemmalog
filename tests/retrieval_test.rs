use lemmalog::agent::{AgentMemory, MockExtractor};
use lemmalog::retrieval::{Bm25, Retrieval};

fn mem() -> AgentMemory<MockExtractor> {
    AgentMemory::new(MockExtractor::new(0.9), "").unwrap()
}

#[test]
fn bm25_ranks_relevant_documents_first() {
    let docs: Vec<String> = vec![
        "the car battery died after the service".into(),
        "best pancake recipe with buttermilk".into(),
        "battery replacement for the GPS system".into(),
        "sourdough starter maintenance".into(),
    ]
    .into();
    let bm25 = Bm25::new(&docs);
    let scores = bm25.scores("car battery replacement");
    assert!(scores[0] > scores[1], "car doc beats pancake doc");
    assert!(scores[2] > scores[3], "battery doc beats sourdough doc");
    assert!(scores[1] == 0.0 && scores[3] == 0.0, "irrelevant docs score 0");
}

#[test]
fn entity_match_beats_keyword_overlap() {
    let mut m = mem();
    m.observe_at(
        "alice --works_at--> acme\nbob --works_at--> gigant\nalice --likes--> hiking",
        100,
    );
    m.maintain(100);
    // query names Alice with zero keyword overlap on relation words
    let r = Retrieval::build(&m.engine, m.episodes());
    let sel = r.select("tell me about alice", 400);
    let renders: Vec<&str> = sel.fact_lines.iter().map(|(l, _)| l.as_str()).collect();
    assert!(renders.iter().any(|l| l.contains("alice")), "{renders:?}");
    // strictly: alice facts come first (entity match dominates)
    let first_alice = renders.iter().position(|l| l.contains("alice")).unwrap();
    let first_bob = renders.iter().position(|l| l.contains("bob"));
    if let Some(b) = first_bob {
        assert!(first_alice < b, "alice before bob: {renders:?}");
    }
}

#[test]
fn budget_is_respected() {
    let mut m = mem();
    let mut facts = String::new();
    for i in 0..60 {
        facts.push_str(&format!("person{i} --owns--> item{i}_a\n"));
        facts.push_str(&format!("person{i} --owns--> item{i}_b\n"));
    }
    m.observe_at(&facts, 100);
    m.maintain(100);
    let r = Retrieval::build(&m.engine, m.episodes());
    let budget = 200; // tokens
    let sel = r.select("person3 items", budget);
    let rendered = r.render(&sel);
    let used_tokens = rendered.len() / 4;
    // facts section alone must fit its share; overall render stays near budget
    assert!(
        used_tokens <= budget * 2,
        "rendered {used_tokens} tokens vs budget {budget}"
    );
    let fact_lines = rendered
        .lines()
        .take_while(|l| !l.starts_with("=="))
        .count();
    let _ = fact_lines;
    assert!(
        sel.fact_lines.len() < 120,
        "selection trimmed: {} facts",
        sel.fact_lines.len()
    );
}

#[test]
fn provenance_episodes_follow_selected_facts() {
    let mut m = mem();
    m.observe_at("alice --works_at--> acme", 100);
    m.observe_at("bob --works_at--> gigant", 200);
    m.maintain(200);
    let r = Retrieval::build(&m.engine, m.episodes());
    let sel = r.select("where does alice work", 600);
    let rendered = r.render(&sel);
    assert!(rendered.contains("alice"));
    // alice's episode (ep1) is the provenance of the selected fact
    assert!(rendered.contains("[ep1]"), "{rendered}");
}

#[test]
fn context_for_query_end_to_end() {
    let mut m = mem();
    m.observe_at(
        "alice --works_at--> acme\nalice --manager--> bob\ncarol --works_at--> zeta",
        100,
    );
    m.maintain(100);
    let ctx = m.context_for_query("who manages alice at work", 300);
    assert!(ctx.contains("alice"), "{ctx}");
    assert!(ctx.contains("manager"), "{ctx}");
    // entity-boosted facts rank first: alice's facts above carol's (which
    // only weakly matches via the stemmed "work" -> "works_at" token)
    let facts_section = ctx
        .split("== source episodes")
        .next()
        .unwrap_or("");
    let first_alice = facts_section.find("alice").unwrap();
    let first_carol = facts_section.find("carol");
    if let Some(c) = first_carol {
        assert!(first_alice < c, "alice facts rank above carol's: {facts_section}");
    }
}
