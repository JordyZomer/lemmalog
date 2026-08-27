use lemmalog::{AgentMemory, HashEmbedder, MockExtractor, SemanticIndex, RELEVANCE_RULES};

#[test]
fn semantic_seeding_ranks_and_diffuses() {
    let mut idx = SemanticIndex::new(HashEmbedder::new(256));
    idx.register("acme", "acme industrial cloud supplier legacy");
    idx.register("gigant", "gigant hyperscale cloud platform kubernetes");
    idx.register("zeta", "zeta analytics dashboard product");

    // the query matches gigant's profile best
    let hits = idx.search("which cloud kubernetes platform", 3);
    assert_eq!(hits[0].0, "gigant", "{hits:?}");
    assert!(hits[0].1 > hits[1].1);

    let mut m = AgentMemory::new(MockExtractor::new(0.9), RELEVANCE_RULES).unwrap();
    m.observe_at("acme --links--> gigant\ngigant --links--> zeta", 100);
    m.maintain(100);

    let seeded = m.seed_mentions(&idx, "q1", "which cloud kubernetes platform", 3);
    assert_eq!(seeded.len(), 3);
    m.maintain(100);

    // relevance diffused two hops with t-norm decay along the path
    let (q, zeta) = (m.engine.sym("q1"), m.engine.sym("zeta"));
    let hits = m.query_near(q, zeta);
    let depth3 = hits
        .iter()
        .find(|(k, _)| k[2] == lemmalog::Value::Int(3))
        .expect("zeta reached at depth 3 via acme->gigant->zeta");
    let sim = seeded[0].1 as f64;
    assert!(
        depth3.1.conf < sim && depth3.1.conf > 0.0,
        "decayed conf {} < seed sim {}",
        depth3.1.conf,
        sim
    );
    assert!(depth3.1.prov.contains("semantic"));
}

#[test]
fn registered_profiles_refresh() {
    let mut idx = SemanticIndex::new(HashEmbedder::new(64));
    idx.register("x", "alpha beta");
    idx.register("x", "gamma delta");
    assert_eq!(idx.search("gamma", 1)[0].0, "x");
}
