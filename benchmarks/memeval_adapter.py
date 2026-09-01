"""lemmalog: stratified Datalog agent memory (Rust engine via lemmalog-bench).

Ingest shells out once per conversation (extraction is file-cached by
episode content hash); every question is a `context` call (hybrid BM25 +
graph + embedding retrieval over the snapshot) answered by the
standardized reader, with a `recall` fallback (one targeted re-read of
the BM25-top episodes) when the structured store returns nothing.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from agents_memory.systems._helpers import _qa_results

BIN = os.environ.get(
    "LEMMALOG_BENCH",
    "/Users/jordy/brainlog/target/release/lemmalog-bench",
)
CACHE = os.environ.get("LEMMALOG_CACHE_DIR", "/tmp/locomo-cache-v2")
SNAPS = os.environ.get("LEMMALOG_SNAP_DIR", "/tmp/locomo-snaps-v2")

SYSTEM_INFO = {
    "architecture": (
        "LLM extraction to Datalog triples + stratified seminaive "
        "materialization; hybrid BM25/entity-boost/embedding retrieval"
    ),
    "infrastructure": "lemmalog-bench (Rust), local nomic embeddings",
}

READER_PROMPT = (
    "Answer the question using ONLY the provided context. Work in this "
    "order: (1) Check WHO — find the facts about the question's topic and "
    "verify they are about the person or thing the question asks about. "
    "If the facts concern someone else or the premise misattributes, "
    "reply exactly: Not mentioned. A reference in the question that "
    "simply is not in memory (a comparison point, an anchor like 'before "
    "I got X') does NOT make the premise false when facts about the "
    "actual subject exist — answer from those. (2) If the subject checks "
    "out and any fact about it is present, ANSWER from the evidence — "
    "count, compare, or combine facts and dates as needed — even if the "
    "answer is not stated directly or the evidence seems incomplete. "
    "Refusing step 2 is not allowed once step 1 passed. Reply with just "
    "the answer and nothing else: for how-many questions the bare number "
    "('2', never '2 (…)'); names without qualifiers; dates as natural "
    "words (7 May 2023); no parentheses, no explanation, no lists "
    "unless asked."
)


def _ensure_snapshot(conv: dict) -> Path:
    sid = str(conv.get("sample_id", "unknown"))
    snap_dir = Path(SNAPS)
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap = snap_dir / f"{sid}.snap"
    if snap.exists():
        return snap
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(conv, f)
        conv_path = f.name
    env = dict(os.environ, LEMMALOG_CACHE_DIR=CACHE)
    res = subprocess.run(
        [BIN, "ingest", conv_path, str(snap)],
        capture_output=True, text=True, env=env,
    )
    os.unlink(conv_path)
    if res.returncode != 0:
        raise RuntimeError(f"ingest failed: {res.stderr[-2000:]}")
    return snap


def _reader(client: OpenAI, model: str, question: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": READER_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=120,
    )
    return (resp.choices[0].message.content or "").strip()


REFUSAL_MARKERS = (
    "not mentioned", "no info", "not specified", "no direct",
    "not available", "no evidence", "not found",
)


def _is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    snap = _ensure_snapshot(conv)

    def answer_fn(question: str) -> str:
        ctx = subprocess.run(
            [BIN, "context", str(snap), question],
            capture_output=True, text=True,
        ).stdout
        answer = _reader(client, llm_model, question, ctx)
        if answer and _is_refusal(answer):
            # retry once when the store genuinely holds topic evidence:
            # the check matches relation/object content (not subject
            # names), so misattributed premises fail it and stay refused
            ev = subprocess.run(
                [BIN, "hasevidence", str(snap), question],
                capture_output=True, text=True,
            ).stdout
            if ev.strip():
                answer = _reader(
                    client, llm_model, question,
                    ctx + "\nVERIFIED FACTS FROM MEMORY on this question's topic "
                    "(the subject HAS facts in the store — answer from them if "
                    "they are about the person the question asks about):\n" + ev,
                )
        if not answer:
            # structured store had nothing: one targeted re-read of the
            # most relevant source episodes
            extra = subprocess.run(
                [BIN, "recall", str(snap), question],
                capture_output=True, text=True,
            ).stdout
            if extra.strip():
                answer = _reader(client, llm_model, question, extra)
        return answer

    return _qa_results(
        conv, answer_fn, run_judge,
        category_names=category_names, judge_fn=judge_fn,
    )
