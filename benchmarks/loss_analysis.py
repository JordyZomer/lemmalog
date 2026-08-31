#!/usr/bin/env python3
"""Loss accounting for lemmalog benchmark runs.

For every question scoring below threshold, traces where the answer broke:
  refusal       — reader refused (gold non-empty)
  extraction    — gold tokens absent from FACT records, present in EP records
  retrieval     — gold tokens in FACTs but missing from the assembled context
  reader        — gold tokens present in context, answer still wrong
  format        — partial credit 0.3-0.6: semantically close, metric mismatch
Buckets are ranked by total F1 left on the table per category.
"""
import json, re, subprocess, sys, os
from collections import defaultdict

BENCH = "/Users/jordy/brainlog/target/release/lemmalog-bench"
STOP = set("a an the at in on of to is are was were be do does did who what which for with and i my me you your we our it its this that not mentioned unknown".split())

def toks(s, minlen=3):
    s = str(s)
    out = set()
    for t in re.findall(r"[a-z0-9]+", s.lower()):
        if t in STOP:
            continue
        if any(c.isdigit() for c in t):
            if len(t) >= 1:
                out.add(t)  # "2", "200" are real golds, not noise
        elif len(t) >= minlen:
            out.add(t)
    return out

def snapshot_parts(path):
    facts, eps = [], []
    for line in open(path, errors="replace"):
        if line.startswith("FACT\t"):
            facts.append(line)
        elif line.startswith("EP\t"):
            eps.append(line)
    return "".join(facts).lower(), "".join(eps).lower()

def classify(row, snap_dir):
    sid, gold, pred, f1 = row["sample_id"], row["ground_truth"], row["predicted"], row["f1"]
    snap = os.path.join(snap_dir, f"{sid}.snap")
    if not os.path.exists(snap):
        return "no-snapshot", set()
    gt = toks(gold)
    refused = "not mentioned" in pred.lower() or not pred.strip()
    empty_gold = not gt
    if empty_gold:
        return ("won-refusal" if refused else "premise-accepted"), set()
    if not gt:
        return "no-gold", set()
    fact_txt, ep_txt = snapshot_parts(snap)
    in_facts = {t for t in gt if t in fact_txt}
    if refused:
        if len(in_facts) >= max(1, len(gt) // 2):
            return "refusal-with-evidence", gt
        if any(t in ep_txt for t in gt):
            return "refusal+extraction-miss", gt
        return "refusal+no-source", gt
    # answered
    ctx = subprocess.run([BENCH, "context", snap, row["question"]],
                         capture_output=True, text=True).stdout.lower()
    in_ctx = {t for t in gt if t in ctx}
    if f1 >= 0.3:
        return "format-partial", gt
    if len(in_facts) >= max(1, len(gt) // 2):
        if len(in_ctx) >= max(1, len(gt) // 2):
            return "reader", gt
        return "retrieval", gt
    if any(t in ep_txt for t in gt):
        return "extraction", gt
    return "no-source", gt

def run(results_file, snap_dir, label, threshold=0.6):
    rows = json.load(open(results_file))["results"]
    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    samples = defaultdict(list)
    for r in rows:
        cat = r["category_name"]
        if r["f1"] >= threshold:
            continue
        bucket, _ = classify(r, snap_dir)
        agg[cat][bucket] += 1.0 - r["f1"]
        counts[cat][bucket] += 1
        if len(samples[(cat, bucket)]) < 3:
            samples[(cat, bucket)].append((r["question"][:60], str(r["ground_truth"])[:40], str(r["predicted"])[:40]))
    total = defaultdict(float)
    print(f"\n=== {label}: F1 left on the table by bucket (loss = sum of 1-F1, f1<{threshold}) ===")
    for cat in sorted(agg):
        parts = ", ".join(f"{b}={agg[cat][b]:.1f}({counts[cat][b]})" for b in sorted(agg[cat], key=agg[cat].get, reverse=True))
        print(f"  {cat:28s} {parts}")
        for b in agg[cat]:
            total[b] += agg[cat][b]
    print("  TOTAL: " + ", ".join(f"{b}={total[b]:.1f}" for b in sorted(total, key=total.get, reverse=True)))
    with open(f"/tmp/loss_samples_{label}.json", "w") as f:
        json.dump({f"{c}|{b}": s for (c, b), s in samples.items()}, f, indent=1)
    print(f"  samples -> /tmp/loss_samples_{label}.json")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "lme"
    if which == "lme":
        run("/tmp/MemEval/data/lemmalog_longmemeval_s_stratified_102_gpt-41_20260828_132150_results.json",
            "/tmp/lme-snaps-v3", "LME")
    else:
        run("/tmp/MemEval/data/lemmalog_locomo_gpt-41-mini_20260828_133617_results.json",
            "/tmp/locomo-snaps-v2", "LoCoMo")
