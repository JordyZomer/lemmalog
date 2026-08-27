#![cfg(feature = "llm")]

//! Feature-gated: `cargo test --features llm`

use lemmalog::agent::{Episode, Extractor};
use lemmalog::llm::{strip_think, LlmClientExtractor, OpenAiClient};

fn ok_transport(content: &str) -> Box<dyn Fn(&str) -> Result<String, String> + Send> {
    let content = content.to_string();
    Box::new(move |_body| {
        Ok(serde_json::json!({
            "choices": [{"message": {"content": content}}]
        })
        .to_string())
    })
}

#[test]
fn chat_strips_reasoning_artifacts() {
    let c = OpenAiClient::with_transport(ok_transport("<think>hmm</think>Alice --works_at--> Acme"), "m");
    let out = c.chat("sys", "usr").unwrap();
    assert_eq!(out, "Alice --works_at--> Acme");

    let c2 = OpenAiClient::with_transport(
        ok_transport("prefix <think>chain
of thought</think> answer"),
        "m",
    );
    assert_eq!(c2.chat("s", "u").unwrap(), "prefix  answer");

    assert_eq!(strip_think("<think>only thinking, no close"), "");
    assert_eq!(strip_think("clean answer"), "clean answer");
}

#[test]
fn extractor_uses_transport_and_memoizes() {
    let c = OpenAiClient::with_transport(ok_transport("alice --works_at[0.7]--> acme"), "m");
    let mut x = LlmClientExtractor::new(c);
    let ep = Episode {
        id: "ep1".into(),
        text: "natural language".into(),
        ts: 1,
        speaker: None,
    };
    let facts = x.extract(&ep);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].obj, "acme");
    assert!((facts[0].confidence - 0.7).abs() < 1e-9);
    // memoized: same episode never re-calls the model
    let again = x.extract(&ep);
    assert_eq!(again.len(), 1);
    assert_eq!(x.calls, 1);
}

#[test]
fn reasoning_content_fallback() {
    let t: Box<dyn Fn(&str) -> Result<String, String> + Send> = Box::new(|_| {
        Ok(serde_json::json!({
            "choices": [{"message": {"content": "", "reasoning_content": "bob --manager--> carol"}}]
        })
        .to_string())
    });
    let c = OpenAiClient::with_transport(t, "m");
    assert_eq!(c.chat("s", "u").unwrap(), "bob --manager--> carol");
}

#[test]
fn strict_parser_drops_reasoning_leakage() {
    use lemmalog::agent::parse_protocol_strict;
    let leaked = "\
- Acme_Corp --has_products--> products? maybe not because generic. But relation has_products...
- speaker --likes--> Acme_Corp_products? But they said skip opinions, so maybe not.
Acme_Corp --is_a--> brand (from both brands)?
Alice --works_at[0.7]--> Acme Corp
speaker --works_at--> Gigant Systems";
    let facts = parse_protocol_strict(leaked, 0.9);
    // only the one clean, named-entity triple survives
    assert_eq!(facts.len(), 1, "{facts:?}");
    assert_eq!(facts[0].subj, "Alice");
    // 'speaker' subjects are dropped by downstream policy anyway; the
    // strict parser's job is prose/questions/bullets
    let clean = "Alice --manager--> Bob\nBob --manager--> Carol";
    assert_eq!(parse_protocol_strict(clean, 0.9).len(), 2);
}

#[test]
fn longmemeval_scoring() {
    use lemmalog::longmemeval::score_f1;
    let (f1, em) = score_f1("GPS system not functioning correctly", "the GPS system is not functioning correctly");
    assert!(f1 > 0.85, "{f1}");
    assert!(em, "stopword-stripped equality");
    let (f1, em) = score_f1("Samsung Galaxy S22", "iPhone 13");
    assert_eq!((f1, em), (0.0, false));
    let (f1, _) = score_f1("a bike", "bike");
    assert!(f1 > 0.9, "{f1}");
}

#[test]
fn longmemeval_loader_parses_fixture() {
    use lemmalog::longmemeval::{load, parse_date};
    assert_eq!(parse_date("2023/04/10 (Mon) 17:50"), parse_date("2023/04/10 (Mon) 17:50"));
    assert!(parse_date("2023/05/01 (Mon) 09:00") > parse_date("2023/04/10 (Mon) 17:50"));
    let dir = std::env::temp_dir().join("lemmalog-lme");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("fixture.json");
    std::fs::write(
        &path,
        r#"[{"question_id":"q1","question_type":"temporal-reasoning","question":"Q?","answer":"GPS system not functioning correctly","question_date":"2023/04/10 (Mon) 23:07","haystack_dates":["2023/04/10 (Mon) 14:47","2023/04/10 (Mon) 17:50"],"haystack_session_ids":["s2","s1"],"haystack_sessions":[[{"role":"user","content":"later session"}],[{"role":"user","content":"earlier session"}]]}]"#,
    )
    .unwrap();
    let inst = load(path.to_str().unwrap()).unwrap();
    assert_eq!(inst.len(), 1);
    assert_eq!(inst[0].sessions.len(), 2);
    // chronological ordering: earlier first, ids matched via parallel arrays
    assert_eq!(inst[0].sessions[0].id, "s2");
    assert_eq!(inst[0].sessions[0].messages[0].content, "later session");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn chunked_extraction_unions_windows() {
    use lemmalog::agent::{Episode, Extractor};
    use lemmalog::llm::EXTRACT_CHUNK_TARGET;
    use std::sync::{Arc, Mutex};

    // transport answers per part; each part yields its own fact
    let seen_parts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen_parts.clone();
    let t: Box<dyn Fn(&str) -> Result<String, String> + Send> = Box::new(move |body| {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let user = v["messages"][1]["content"].as_str().unwrap().to_string();
        sink.lock().unwrap().push(user.clone());
        let reply = if user.contains("part 1") {
            "the_user --owns--> kayak"
        } else {
            "the_user --owns--> tent"
        };
        Ok(serde_json::json!({"choices": [{"message": {"content": reply}}]}).to_string())
    });
    let c = OpenAiClient::with_transport(t, "m");
    let mut x = LlmClientExtractor::new(c);

    // build an episode long enough to chunk, with message markers
    let mut text = String::from("Session s1 on 2023/01/01:\nuser: I own a kayak.\n");
    while text.len() < EXTRACT_CHUNK_TARGET + 200 {
        text.push_str("assistant: Great, noted for our conversation.\n");
    }
    text.push_str("user: I also own a tent.\n");
    let ep = Episode {
        id: "ep1".into(),
        text,
        ts: 1,
        speaker: Some("the_user".into()),
    };
    let facts = x.extract(&ep);
    assert_eq!(facts.len(), 2, "facts from both chunks unioned: {facts:?}");
    let calls = {
        let s = seen_parts.lock().unwrap();
        s.len()
    };
    assert!(calls >= 2, "chunked into multiple calls: {calls}");
    assert_eq!(x.calls, calls);
    // memoized: re-extraction makes no calls
    let again = x.extract(&ep);
    assert_eq!(again.len(), 2);
    assert_eq!(x.calls, calls);
}

#[test]
fn subject_question_matching_is_whole_token() {
    use lemmalog::longmemeval::subject_matches_question;
    let q = "Which device did I get first, the Samsung Galaxy S22 or the Dell XPS 13?";
    assert!(subject_matches_question("Samsung Galaxy S22", q));
    assert!(subject_matches_question("Dell XPS 13", q));
    assert!(!subject_matches_question("Rachel", q), "unrelated subject: no shared token");
    assert!(!subject_matches_question("art", q), "short/stopword-adjacent tokens don't match");
    // the old bug: two-way raw substring pulled in everything via words
    // like "the"/"did" — whole-token matching must not
    assert!(!subject_matches_question("theater", q), "'the' + substring must not match 'theater'");
}

#[test]
fn file_cached_extractor_roundtrip() {
    use lemmalog::agent::{Episode, Extractor};
    use lemmalog::llm::{LlmClientExtractor, OpenAiClient};
    use lemmalog::longmemeval::OPEN_VOCAB_PROMPT;
    use std::sync::{Arc, Mutex};

    let calls = Arc::new(Mutex::new(0usize));
    let sink = calls.clone();
    let t: Box<dyn Fn(&str) -> Result<String, String> + Send> = Box::new(move |_body| {
        *sink.lock().unwrap() += 1;
        Ok(serde_json::json!({
            "choices": [{"message": {"content": "alice --works_at--> acme"}}]
        })
        .to_string())
    });
    let c = OpenAiClient::with_transport(t, "m");
    let extractor = LlmClientExtractor::new(c).with_prompt(OPEN_VOCAB_PROMPT);
    let dir = std::env::temp_dir().join("lemmalog-cache-test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut cached = extractor.file_cached(dir.to_str().unwrap());

    let ep = Episode {
        id: "ep1".into(),
        text: "I work at acme now".into(),
        ts: 1,
        speaker: Some("alice".into()),
    };
    let f1 = cached.extract(&ep);
    assert_eq!(f1.len(), 1);
    assert_eq!(*calls.lock().unwrap(), 1, "first extraction hits the model");

    // second extraction of the same episode: served from cache, zero calls
    let f2 = cached.extract(&ep);
    assert_eq!(f2.len(), 1, "cache round-trips the fact");
    assert_eq!(*calls.lock().unwrap(), 1, "no additional model call");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn date_objects_normalize_to_comparable_ints() {
    use lemmalog::longmemeval::date_to_int;
    assert_eq!(date_to_int("2019"), Some(20190000));
    assert_eq!(date_to_int("2019-03"), Some(20190300));
    assert_eq!(date_to_int("2019-03-14"), Some(20190314));
    // ordering across granularities holds
    assert!(date_to_int("2019").unwrap() < date_to_int("2019-03").unwrap());
    assert!(date_to_int("2019-03").unwrap() < date_to_int("2019-03-14").unwrap());
    assert!(date_to_int("2019-03-14").unwrap() < date_to_int("2020-01").unwrap());
    // not dates: natural language, ids, partial junk
    assert_eq!(date_to_int("last week"), None);
    assert_eq!(date_to_int("2019-3"), None);
    assert_eq!(date_to_int("12345"), None);
    assert_eq!(date_to_int("20190314"), None);
}
