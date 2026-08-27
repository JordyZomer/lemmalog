use lemmalog::run_eval;

/// The synthetic long-horizon suite: the deterministic memory behaviors
/// LongMemEval shows frontier models fail (knowledge updates, temporal
/// projection, multi-hop reachability, conflict abstention) must all be
/// exact — they are rule-derived, not guessed. Also asserts the token
/// economics the design predicts: bounded assembled context vs linear
/// transcript.
#[test]
fn synthetic_long_horizon_suite() {
    // 500 turns: SCC-correct scoped recompute fully rebuilds dependent
    // closures on supersession (the old same-stratum skip was the latent
    // bug the canonicalization tests exposed); this size keeps the
    // regression suite fast while exercising every behavior
    let rep = run_eval(42, 30, 8, 500, 40);
    assert_eq!(rep.employment_correct, rep.employment_q, "knowledge updates");
    assert!(rep.supersessions > 0, "scenario must exercise supersession");
    assert_eq!(rep.multihop_correct, rep.multihop_q, "multi-hop reachability");
    assert_eq!(rep.abstain_correct, rep.abstain_people, "conflict abstention");
    assert_eq!(rep.accuracy(), 1.0);
    assert!(
        rep.token_savings() > 3.0,
        "context {} vs transcript {} tokens",
        rep.context_tokens,
        rep.transcript_tokens
    );
    // same seed, identical report: deterministic
    let rep2 = run_eval(42, 30, 8, 500, 40);
    assert_eq!(rep.employment_correct, rep2.employment_correct);
    assert_eq!(rep.context_tokens, rep2.context_tokens);
}

#[test]
fn distinct_seeds_distinct_scenarios() {
    let a = run_eval(12345, 20, 6, 300, 20);
    let b = run_eval(54321, 20, 6, 300, 20);
    // different event streams produce different reports (at least one
    // metric must differ; counts can coincide by chance)
    let differs = a.supersessions != b.supersessions
        || a.employment_q != b.employment_q
        || a.multihop_q != b.multihop_q
        || a.abstain_people != b.abstain_people
        || a.transcript_tokens != b.transcript_tokens;
    assert!(differs, "seeds produced identical reports: {a:?} vs {b:?}");
    // and both remain fully correct
    assert_eq!(a.accuracy(), 1.0);
    assert_eq!(b.accuracy(), 1.0);
}
