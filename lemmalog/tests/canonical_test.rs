use lemmalog::canonical::{alias_conflicts, assert_alias, install_canonicalization};
use lemmalog::{Ann, Engine, Value};

fn syms(e: &mut Engine, xs: &[&str]) -> Vec<Value> {
    xs.iter().map(|x| e.sym(x)).collect()
}

fn seed(e: &mut Engine) {
    // two raw spellings of the same employer + two unrelated entities
    for (s, r, o) in [
        ("alice", "works_at", "Acme Corp"),
        ("bob", "works_at", "Acme_Corp"),
        ("carol", "works_at", "Gigant Systems"),
        ("alice", "likes", "hiking"),
    ] {
        let v = syms(e, &[s, r, o]);
        e.declare("current", &v, Ann::unit());
    }
}

#[test]
fn star_aliasing_closure_and_views() {
    let mut e = Engine::new();
    seed(&mut e);
    install_canonicalization(&mut e, &["current"]).unwrap();
    assert_alias(&mut e, "Acme_Corp", "Acme Corp", 0.9);
    e.run();

    // same_as is reflexive for entity-seeded names and closed over aliases
    let (acme, acme_ugly) = (e.sym("Acme Corp"), e.sym("Acme_Corp"));
    assert_eq!(e.query("same_as", &[Some(acme), Some(acme)]).len(), 1, "reflexive");
    assert_eq!(e.query("same_as", &[Some(acme_ugly), Some(acme)]).len(), 1);
    assert_eq!(e.query("same_as", &[Some(acme), Some(acme_ugly)]).len(), 1, "symmetric");

    // canonical view collapses spellings; unrelated entities keep own identity
    let canon = e.query("current_canon", &[]);
    let employers: Vec<Vec<Value>> = canon.iter().map(|(k, _)| k.clone()).collect();
    assert_eq!(employers.len(), 4, "all facts survive the view: {employers:?}");
    let alice_canon: Vec<_> = canon
        .iter()
        .filter(|(k, _)| k[0] == e.sym("alice") && k[1] == e.sym("works_at"))
        .map(|(k, _)| k[2])
        .collect();
    assert_eq!(alice_canon, vec![e.sym("Acme Corp")], "alice projected to canonical");

    // confidence propagates through the closure (product of the one edge)
    let sa = e.query("same_as", &[Some(acme_ugly), Some(acme)]).remove(0);
    assert!((sa.1.conf - 0.9).abs() < 1e-9);

    // counting over the canonical view is stable across spellings
    e.install_program("employer_headcount(O, count(P)) :- current_canon(P, works_at, O).")
        .unwrap();
    e.run();
    let acme_hc = e.query("employer_headcount", &[Some(acme), None]).remove(0);
    assert_eq!(acme_hc.0[1], Value::Int(2), "both spellings counted once, canonically");
}

#[test]
fn two_hop_confidence_products() {
    let mut e = Engine::new();
    seed(&mut e);
    install_canonicalization(&mut e, &["current"]).unwrap();
    assert_alias(&mut e, "Acme_Corp", "Acme Corp HQ", 0.8);
    assert_alias(&mut e, "Acme Corp", "Acme Corp HQ", 0.9);
    e.run();
    let (ugly, hq) = (e.sym("Acme_Corp"), e.sym("Acme Corp HQ"));
    let sa = e.query("same_as", &[Some(ugly), Some(hq)]).remove(0);
    // direct edge 0.8 vs two-hop 0.9*... the strongest witness merges to max
    assert!(sa.1.conf >= 0.8 - 1e-9, "conf {}", sa.1.conf);
}

#[test]
fn conflicts_escalate_instead_of_merging() {
    let mut e = Engine::new();
    seed(&mut e);
    install_canonicalization(&mut e, &["current"]).unwrap();
    // one local with two canonicals: split identity
    assert_alias(&mut e, "Acme_Corp", "Acme Corp", 0.9);
    assert_alias(&mut e, "Acme_Corp", "Acme Corporation", 0.9);
    // chain: a name that is both local and canonical
    assert_alias(&mut e, "Acme Corporation", "Acme Corp Global", 0.9);
    e.run();
    let conflicts = alias_conflicts(&e);
    assert!(conflicts.len() >= 2, "{conflicts:?}");
}

#[test]
fn retracting_an_alias_collapses_the_closure() {
    let mut e = Engine::new();
    seed(&mut e);
    install_canonicalization(&mut e, &["current"]).unwrap();
    assert_alias(&mut e, "Acme_Corp", "Acme Corp", 0.9);
    e.run();
    let (ugly, acme) = (e.sym("Acme_Corp"), e.sym("Acme Corp"));
    assert_eq!(e.query("same_as", &[Some(ugly), Some(acme)]).len(), 1);

    // undo the merge: read-side canonicalization is reversible
    e.retract("alias", &[ugly, acme]);
    e.run();
    assert_eq!(
        e.query("same_as", &[Some(ugly), Some(acme)]).len(),
        0,
        "closure collapses after alias retraction"
    );
    // the view falls back to raw facts (reflexivity keeps everything)
    assert_eq!(e.query("current_canon", &[]).len(), 4);
}
