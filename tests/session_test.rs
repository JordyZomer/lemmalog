use lemmalog::session::Session;

#[test]
fn repl_session_end_to_end() {
    let mut s = Session::new();
    for line in [
        "rule current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.",
        "rule reports_to(X,Y) :- current(X,\"manager\",Y).",
        "rule trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).",
        "+ edge(alice, manager, bob, 0, MAX, 1) @0.9 #ep1",
        "+ edge(bob, manager, carol, 0, MAX, 1) @0.9 #ep2",
        "now 10",
        "run",
    ] {
        let out = s.execute(line);
        assert!(!out.starts_with("error"), "{line} -> {out}");
    }
    assert!(s.execute("? reports_to(\"alice\", Y)").contains("Y=carol"));
    assert!(s.execute("?? reports_to(\"alice\", Y)").contains("Y=bob"));
    let why = s.execute("why reports_to(alice, carol)");
    assert!(why.contains("via trans") && why.contains("ep1"));
    // one witness per rule label: no duplicate trans blocks
    assert_eq!(why.matches("via trans").count(), 1, "{why}");
    let dump = s.execute("dump reports_to");
    assert!(dump.contains("prov [\"ep1\", \"ep2\"]"));
    // batch registry + uninstall
    assert!(s.execute("batches").contains("b0"));
    assert!(s.execute("rm b2").contains("uninstalled b2"));
    s.execute("run");
    // trans rule gone: only the two direct facts remain
    assert!(s.execute("? reports_to(\"alice\", \"carol\")").contains("(no answers)"));
    assert!(!s.execute("? reports_to(\"alice\", \"bob\")").is_empty());
    assert!(s.execute("rm b1").contains("uninstalled b1"));
    s.execute("run");
    assert!(s.execute("dump reports_to").contains("(empty)"));
    // error paths are strings, not panics
    assert!(s.execute("rule p(X) :- ").starts_with("error:"));
    assert!(s.execute("+ p(X)").starts_with("error:"));
    assert!(s.execute("bogus").starts_with("error:"));
}
