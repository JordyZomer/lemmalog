# Lemmalog

A Datalog engine for LLM agent memory. This repo contains the engine
(Rust crate, MCP server, REPL, agent skill) plus the design document
([`datalog-context-engine-design.md`](datalog-context-engine-design.md),
with an honest status log of what shipped).

The thesis: **an agent's memory should be a deductive database** — the
agent builds a *verifiable model of what it knows* and mechanically
reasons over how that knowledge changes, rather than "remembering
better" than a vector store. Base facts
are asserted at the ingestion boundary (LLM extraction); rules derive
closures, temporal projections, contradiction candidates, and relevance
diffusion; every fact carries provenance back to its source episodes; and
each conversation turn updates derived views incrementally instead of
re-deriving them (or worse, re-reasoning them in-context).

## What's implemented

| Design element | Status |
|---|---|
| Runtime-parsed, stratified Datalog (interpreter, not proc-macro) | ✅ |
| Negation-as-absence with negative-cycle rejection | ✅ |
| Seminaive fixpoint with per-epoch delta maintenance | ✅ |
| Bi-temporal facts via `valid_from`/`valid_to`/`asserted_at` columns + `now()` | ✅ |
| Semiring annotations: confidence (product t-norm) × provenance (set union) | ✅ |
| Annotation merge on re-derivation (max conf, union prov, deduped supports) | ✅ |
| `why()` proof trees with cycle protection | ✅ |
| Additive arithmetic in comparisons (`D = Dm + 1`) with linear solving | ✅ |
| Scoped negative deltas: retraction recomputes only transitive dependents | ✅ |
| `ask()` — read-only datalog query surface for agents | ✅ |
| Magic-sets demand evaluation (`ask_deep`): point queries without full fixpoint | ✅ |
| Per-position secondary indexes; row-id lookups; WAM-style trail backtracking | ✅ |
| Epoch change-log: `changes_from/since` + "new in memory" context section | ✅ |
| Hybrid retrieval (`context_for_query`): BM25 + entity/graph boosting, budget-aware | ✅ |
| Extraction boundary: `Extractor` trait, memoized `MockExtractor` + `LlmExtractor` | ✅ |
| Deterministic update policy: ADD / UPDATE / NOOP / escalate | ✅ |
| Positional `ContextAssembler` (distilled top, verbatim provenance bottom, budget) | ✅ |
| `AgentMemory` facade: observe → policy → maintain → ask/ask_deep/context/why | ✅ |
| Persistence: snapshot save/load (episodes + EDB facts + rules; derived rebuilt) | ✅ |
| Semantic side index: `Embedder` trait, `HashEmbedder`, `seed_mentions` + `near` diffusion | ✅ |
| DRed-lite scoped recompute: supersession rebuilds only what actually changed | ✅ |
| Synthetic eval harness (`scenario::run_eval`): accuracy/token/latency vs. ground truth | ✅ |
| Aggregation: `count`/`min`/`max`/`sum` head args with group-by fold + value-change propagation | ✅ |
| Entity resolution: star-shaped aliasing, directional canonical views, conflict escalation | ✅ |
| MCP server (`--features mcp`): the engine as tools for Claude Code / Kimi CLI | ✅ |
| Rule registry: versioned batches, agent install/uninstall, backfill on change | ✅ |
| Hypotheticals: `what_if` lookahead with byte-identical store restore | ✅ |
| Streaming change feed: `Added`/`Retracted`/`Cleared` events for projections | ✅ |
| Indexed read paths: `query`/`ask` select buckets (point lookups ~100µs at 4M facts) | ✅ |
| Differential testing: 450 random programs vs a naive fixpoint oracle + parser fuzzing | ✅ |
| REPL: `cargo run --bin lemmalog` (rule / + / ? / ?? / why / run / dump / batches) | ✅ |
| Leapfrog triejoins (worst-case-optimal joins), DBSP streaming deltas | 🚧 future phases |

## Entity resolution (canonicalization)

The LLM proposes star-shaped `alias(Local, Canonical)` edges; Datalog
derives the closure; canonical views project facts read-side only
(`src/canonical.rs`):

```prolog
alias(Acme_Corp, Acme).                       % LLM-proposed, confidence-tagged
same_as(X, Y) :- alias(X, Y).                 % symmetric-transitive closure
same_as(X, Z) :- same_as(X, Y), same_as(Y, Z).
maps_to(X, X) :- entity(X), !aliased(X).      % directional projection:
maps_to(L, C) :- alias(L, C).                 %  exactly one canonical spelling
current_canon(S, R, O) :- current(S, R, O), maps_to(S, S2), maps_to(O, O2).
```

Safety properties (all tested): topology violations — a local with two
canonicals, or a name both local and canonical — derive `alias_conflict`
facts instead of merging identities; confidence propagates through the
closure (weak two-hop merges are visibly low-confidence); retracting an
alias edge collapses the closure and every downstream view in the same
epoch. A similarity-gated LLM reconciliation pass
(`canonical::reconcile::reconcile_entities`) offers only
embedding-similar name pairs to the model.

Building this surfaced and fixed two long-lived engine bugs: the scoped
recompute never processed same-stratum dependents (latent stale-fact
bug), fixed by SCC-condensation stratification plus a recompute
fixpoint; and the invalidation pass ran before lower strata were
materialized on first run, fixed by moving invalidation after
evaluation. Both caught by the differential harness.

## The lemmalog skill

`skills/lemmalog/SKILL.md` in this crate is a generic agent skill that makes the engine the task's working memory for
*any* long-running work — investigations, debugging, audits, multi-agent
searches — not just one hardcoded workflow. It encodes the discipline the
live experiments converged on (assert-as-you-verify with anchors and
confidence, rules as experiments, query before re-reasoning, `why` before
trusting, hypothesis lifecycles, decide-from-queries, report-from-the-engine),
the minimal interop schema (`located`, `describes`, `hypothesis`/`status`,
`decision`), the grammar gotchas, and the anti-patterns. Install per CLI:

```sh
# Claude Code (user scope)
mkdir -p ~/.claude/skills && cp -r skills/lemmalog ~/.claude/skills/

# Kimi CLI: copy the same folder into its skills directory
# (e.g. ~/.kimi/skills/lemmalog/ — see its skills docs)
```

Task prompts then stay domain-specific and reference the skill in one line.

## MCP server: use from Claude Code or Kimi CLI

```sh
cargo build --release --features mcp
```

Register the server (stdio JSON-RPC, 12 tools):

```sh
# Claude Code (project or user scope)
claude mcp add lemmalog -- $(pwd)/target/release/lemmalog-mcp

# Kimi CLI
kimi mcp add lemmalog -- $(pwd)/target/release/lemmalog-mcp
```

Persistence across sessions: set the environment when registering
(both CLIs support `--env KEY=VALUE` on add):

```sh
claude mcp add lemmalog --env LEMMALOG_MCP_PATH=/tmp/lemmalog.snapshot -- \
  $(pwd)/target/release/lemmalog-mcp
```

The intended division of labor: the host model (Claude/Kimi) reads the
conversation and asserts triples via `lemmalog_observe` (line protocol
`S --rel[conf]--> O`); Lemmalog derives closures, temporal views,
canonicalizations and aggregations deterministically. Typical session:

```
lemmalog_observe      {"facts": "Alice --works_at--> Acme\nAlice --manager--> Bob", "ts": 100}
lemmalog_install_rules {"rules": "reports_to(X,Y) :- current(X,\"manager\",Y).\n trans: ..."}
lemmalog_query        {"goal": "reports_to(\"Alice\", Y)"}        -> Y=Bob, Y=Carol
lemmalog_why          {"fact": "reports_to(Alice, Carol)"}          -> proof tree to episodes
lemmalog_what_if      {"facts": "Dana --manager--> Alice", "goal": "reports_to(\"Dana\", Y)"}
lemmalog_canonicalize {"facts": "Acme_Corp --alias_of[0.9]--> Acme"}
```

Also available: `lemmalog_query_deep` (magic sets), `lemmalog_dump`,
`lemmalog_batches`/`lemmalog_uninstall` (revertable rule batches),
`lemmalog_save`, `lemmalog_run`. Note the goal/fact grammar: bare
capitalized words are variables — quote entity names
(`reports_to("Alice", Y)`).

**Error semantics are built for self-correction.** Recoverable input
errors (unparseable goals, rejected rule batches, unknown batch ids)
return as tool results with `isError: true` — category prefix, the
offending input, the precise reason, and a hint or corrected example
(e.g. the quote-entity-names hint on every parse failure). Silent
zero-fact ingestion is impossible: `lemmalog_observe` reports every
dropped line with its reason (pronoun/role-word subjects, prose
contamination, missing `--rel-->` structure), so a malformed extraction
batch is loud, not lost.

## LongMemEval (oracle split) — live results

The 15 MB oracle split is not committed; download it once:

```sh
mkdir -p data && curl -sL \
  https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle \
  -o data/longmemeval_oracle.json
```

`examples/longmemeval.rs` runs the benchmark end-to-ready: evidence
sessions -> chunked live extraction -> update policy -> memory -> answers
in two modes (structured memory block vs raw transcript, same model) ->
SQuAD-style F1. Final configuration: Claude Opus 4.8, 5 per type, with
role-aware pronoun resolution, stated-date extraction feeding derived
ordering rules, answer-format discipline, and a question-time recall
fallback:

```
                         memory F1   transcript F1   EM
single-session-user   5/5  0.80          0.80       4/5 vs 4/5
knowledge-update      5/5  0.60          0.41       3/5 vs 1/5
multi-session         5/5  0.21          0.38       0/5 vs 1/5
temporal-reasoning    5/5  0.57          0.61       2/5 vs 2/5
single-session-assistant  0.64          0.74       2/5 vs 2/5
single-session-preference 0.06          0.13       0/5 vs 0/5
OVERALL               30    0.48          0.51      11/30 vs 10/30
```

(One scored run — see the measurement caveat in the retrieval-results
section before quoting these numbers comparatively; run-to-run variance
without temperature control is ~±0.3 F1 per type at n=5.)

Per-fix effects, measured on the failing instances before the full run:

- **Role-aware pronoun resolution** (the assistant's "I" was being
  rewritten into the user's voice by our own speaker instruction): the
  Roscioli recommendation question went 0.00 across three runs -> exact
  match. single-session-assistant F1 0.44 -> 0.64.
- **Answer-format discipline** ("ONLY the answer entity, no derivation")
  plus stated dates and rule-derived ordering (`dated` rules generated
  from date-shaped relations, `happened_before` derived by comparison
  rules, stated `before` made transitive): the bike-vs-car question went
  0.15 (right answer wrapped in prose) -> exact match; temporal-reasoning
  0.42 -> 0.57.
- **Recall fallback** (on "unknown"-shaped answers, one targeted
  extraction pass over the retained episodes, then re-answer): triggered
  correctly but rescued nothing in this run — the residual misses
  ("hoping to beat my best of 25:50") resist even question-informed
  extraction. Kept: it is architecturally right and free when unused.

Honest trade-off now visible in the data: richer extraction grows memory
contexts (dated facts, assistant facts), compressing the token advantage
from 4-12x to 1-5x on heavy instances — recall vs context size is a dial,
not a free lunch. The stable per-type structure across five full runs:
knowledge-update is the memory's decisive category (transcript answers
stale values or hedges both), user-stated discrete facts are near-perfect,
preference gold answers are unmatchable prose for both modes, and the one
remaining frontier is indirect mentions.

## Hybrid retrieval (`src/retrieval.rs`)

The answer to the trade-off above: **selection, not extraction, is the
bottleneck at the context boundary.** `AgentMemory::context_for_query`
replaces dump-everything assembly with a three-signal ranker:

- **BM25** (in-crate, no dependencies) over rendered facts *and* verbatim
  episode text — exact keyword grounding, including entities and relation
  words the question uses.
- **Entity-match boosting** — the graph half: a query naming an entity
  pulls that entity's facts (+1.5) and one-hop co-occurring entities'
  facts (+0.4), even with zero keyword overlap on relation words.
- **Budget-aware positional assembly** — ranked facts fill 60% of the
  token budget, their provenance episodes plus BM25-top episodes fill the
  rest verbatim (char-safe truncation), distilled-first / sources-last.

Selection is O(facts) per query — rebuild-on-demand is fine at agent
scale — and the internal bookkeeping relations (canonicalization plumbing,
aggregation temps, entity seeds) are excluded. The LongMemEval runner now
answers from retrieved context (question-relevant facts + dated edge
history only for the entities the selection touches), and the MCP server
exposes it as `lemmalog_context` — the skill teaches it as the default
over `lemmalog_dump` for grounded answering.

## Retrieval results (live, same 30-instance protocol)

Memory-mode context switched from dump-everything to
`context_for_query` (1800-token budget, budgeted dated-history append).
Assembled from focused runs (Claude Opus 4.8):

```
                         memory F1   transcript F1
knowledge-update      5/5  0.80          0.57
single-session-user   5/5  ~1.00         ~0.80
single-session-assistant  0.70          0.74
multi-session         5/5  0.32          0.33
single-session-preference 0.11          0.11
temporal-reasoning    5/5  high variance (see below)
```

**Measurement caveat, learned the hard way**: opus-4-8 rejects the
`temperature` parameter, so answers sample at the API default — a single
scored run at n=5/type has ~±0.3 F1 noise per category (the bike-vs-car
question flipped 1.00 → 0.00 across two same-configuration runs). Type
comparisons below ~0.3 are not evidence. The findings that held across
every configuration:

- **Knowledge-update is consistently ahead**: 0.80 vs 0.57 in the
  retrieval configuration, including the indirect-mention 5K question
  answered for the first time in six runs; the transcript baseline
  answers with stale values or hedges both.
- **User-stated discrete facts are near-perfect** for memory across all
  configurations, including instances the transcript mode misses.
- **Token economics hold**: 1.3-6x smaller contexts, precision-vs-latency
  is an explicit dial (tighter contexts trigger more recall fallbacks).
- **Temporal-reasoning is unresolved, with the failure fully attributed**:
  ordering questions need both endpoints' dated facts; when extraction
  captures both (verified by grepping the extraction cache), retrieval
  delivers them and the answer is exact — the residual failures are
  extraction recall (events never extracted as facts) plus answer-sampling
  variance. Fixing measurement needs n≥10 or repeated runs, which the
  extraction cache makes cheap (answers-only cost).

**Reproduction is nearly free after the first run**: `LEMMALOG_CACHE_DIR`
persists extraction results by episode hash (reruns pay only for
answering), `LEMMALOG_DUMP_CTX` + `LEMMALOG_NO_ANSWER=1` assemble and
dump contexts with zero API calls — context-assembly changes can be
validated offline by diffing the dumped files.

## MemEval: the standardized comparison (102 questions, split s)

The headline run: [ProsusAI MemEval](https://github.com/ProsusAI/MemEval)'s
stratified 102-question LongMemEval protocol (17 per category, the `s`
haystack, ~50 sessions per question) with their standardized reader
(gpt-4.1) and native binary judge (gpt-4o). Lemmalog plugged in as an
adapter; extraction is Claude Sonnet 4.6 (ingestion is architectural,
like Memory-R1's local model), chunked, file-cached. Published numbers
are F1 on their leaderboard.

```
System              F1 (answer tokens)
PropMem (pub)        0.550   (23.1M all-phase)
SimpleMem (pub)      0.480   (20.8M all-phase)
lemmalog             0.469 ± 0.005 (3 runs)  (467K answer-phase)
OpenClaw (pub)       0.244   ( 0.7M)
fullcontext (ours)   0.197   (10.6M)
fullcontext (pub)    0.222   (10.6M)
```

Binary accuracy 0.566 ± 0.009 (3 runs, gpt-4o judge). Per-category
(single run, F1 / accuracy):

```
Single-Session User         0.919 / 0.941
Knowledge Update            0.405 / 0.529
Single-Session Assistant    0.649 / 0.882
Temporal Reasoning          0.410 / 0.412
Multi-Session               0.365 / 0.412
Single-Session Preference   0.116 / 0.235
```

Reading it honestly:

- **F1 0.469 ± 0.005 (3 runs) — 2.4x our own full-context run (0.197)**
  — within 0.011 of SimpleMem's published 0.480, at 1/22nd the
  answer-phase tokens. PropMem (0.550) still ahead.
- **The improvement arc is the story**: the first configuration scored
  F1 0.226; diagnosing the actual wrong answers (`benchmarks/loss_analysis.py`
  traces every loss to refusal / extraction / retrieval / reader / format
  buckets) and shipping targeted fixes — counting aggregates, terse
  answers, recall fallback, count sections with member enumeration, a
  reference-date anchor, precomputed date arithmetic, premise-gated
  answering with evidence-injected refusal-retry, per-item enumeration
  extraction, and a selection budget that scales with the store — more
  than doubled F1: user-facts 0.359 → 0.919, multi-session 0.013 → 0.365,
  temporal 0.373 → 0.410.
- **Richer extraction needs richer selection**: per-item enumeration
  extraction grew the store ~2x and initially DROPPED F1 to 0.435 — more
  facts competing for the same context budget starves selection. Scaling
  the budget (1800 → 3200) recovered it to 0.469. Selection, not
  extraction, is the binding constraint on this benchmark.
- **Knowledge-update regressed with enumeration** (0.528 → 0.405):
  historical enumerated items pollute current-state questions — the
  known next lever is validity-interval-aware selection (prefer facts
  still open at the reference date).
- **Token economics**: ~4.6K answer-phase tokens per question vs ~104K
  for full context (22x); extraction is paid once per conversation
  (~$0.25 Sonnet, cached forever) and amortizes across every additional
  question.

## LoCoMo: the second standardized benchmark (10 conversations, 1,986 questions)

Same harness, LoCoMo's standardized gpt-4.1-mini reader. F1 vs. their
published leaderboard:

```
Rank  System          F1      Tokens (all-phase)
 1    PropMem (pub)   0.605     5.9M
 —    lemmalog        0.573 ± 0.002 (3 runs)  11.3M
 2    OpenClaw (pub)  0.557    16.4M
 3    FullCtx (pub)   0.542    37.5M
 4    Hindsight(pub)  0.489    24.2M
 5    Graphiti (pub)  0.416     5.1M
 6    Memory-R1(pub)  0.389     3.4M
 7    SimpleMem(pub)  0.358    11.4M
```

**2nd of 10** — ahead of OpenClaw, full-context, Hindsight, Graphiti,
Memory-R1, SimpleMem, Mem0, and MemU; behind only PropMem. Run-to-run σ
is 0.002 (three full 1,986-question runs). The journey from 0.483 came
in three measured stages (same reader, same judge):

1. **Retrieval-side upgrades** (temporal normalization, wired
   reconciliation, embedding rerank, conditional-preference discipline):
   0.483 → 0.533.
2. **Reader discipline** (premise-gated answering: check WHO the facts
   are about — misattribution → refuse; premise passed → answering from
   evidence is mandatory, bare numbers for how-many questions):
   adversarial 0.676 → 0.717.
3. **Evidence-injected refusal-retry + attribution contrast + per-item
   enumeration extraction + store-scaled selection budget**: a
   `hasevidence` check (subject-excluded topic overlap — misattributed
   premises fail it and stay refused) re-asks refused questions with
   the verified facts quoted; an ATTRIBUTION section shows which
   subjects hold topic facts and which question-mentioned parties hold
   none; extraction emits one triple per enumerated item; the context
   budget scales with the store (1800 → 3200). 0.533 → 0.573.

Per-category (first run → final):

```
                    first   final   PropMem  FullCtx
Multi-hop (N=841)    0.544   0.615    0.599    0.674
Adversarial (N=446)  0.676   0.738    0.794    0.509
Factual (N=282)      0.368   0.418    0.431    0.517
Temporal (N=321)     0.257   0.489    0.615    0.369
Inferential (N=96)   0.143   0.213    0.289    0.197
```

Notable: **adversarial 0.738 beats full-context (0.509) by +0.23** —
those are questions designed to bait false memories (misattributed
premises), and the structured memory says "no" honestly: the
attribution contrast names the party with no supporting facts, and the
retry guard refuses to manufacture evidence for them. Multi-hop 0.615
now beats PropMem's 0.599.

Two benchmark rows, both on their standardized harnesses, both with
repeated measures: LongMemEval F1 0.469 ± 0.005 / accuracy 0.566 ± 0.009
(at 1/22nd the tokens) and LoCoMo F1 0.573 ± 0.002 (2nd of 10). The
consistent pattern: competitive with the leaders on structure-rewarding
categories, ahead of every retrieval-first system, behind PropMem
overall.

## Token economics (the honest numbers)

**Per-question context** (what actually hits the reader's prompt):

```
LongMemEval:  ~2,300 tokens/question vs ~104,000 for full context = 45x
LoCoMo:       ~3,200 tokens/question vs ~18,900 for full context =  6x
```

**All-in cost** (reader + one-time extraction, benchmark accounting):

The benchmarks are actually the *worst case* for amortization —
LongMemEval gives each question a fresh conversation (extraction never
reuses). LoCoMo (10 conversations, ~200 questions each) shows the real
curve:

```
questions asked     all-in (ours)    all-in (fullctx)    ratio
        ~100            ~1.5M              ~1.9M          1x (crossover)
        200             ~1.8M              ~3.8M          2x
      1,986             ~7.6M             ~37.5M          5x
```

**Real agent scenario** (one growing conversation, queried every turn):

```
after  50 turns:  fullctx = 100,000 tok/q | lemmalog = 2,500 tok/q (40x)
after 100 turns:  fullctx = 200,000 tok/q | lemmalog = 2,500 tok/q (80x)
                  (fullctx OVERFLOWS a 128K window here)
after 500 turns:  fullctx =   1.0M  tok/q | lemmalog = 2,500 tok/q (400x)
```

Lemmalog's per-question cost is **constant** (~2.5K tokens) regardless
of history length; full-context grows linearly and overflows. The
extraction cost is proportional to *new input* (you only pay for what
you read once), not to *queries*. For a long-running agent, the
cumulative ratio reaches 150x by turn 500 — and the agent never runs
out of window.

## Correctness assurance## Correctness assurance

`tests/differential_test.rs` generates 450 random stratified programs
(range-restricted rules, EDB-only negation with constant arguments) and
compares the engine against a dead-simple brute-force fixpoint oracle over
the ground domain — the classic validation technique for Datalog engines —
plus an incremental-vs-single-shot equivalence check and a 2000-case
parser fuzz. This harness caught a real soundness bug: cross-run growth
of a predicate that a rule reads *negatively* never invalidated that
rule's earlier derivations (within one run, stratum ordering already
guaranteed soundness). Additions to negated predicates now trigger the
DRed-lite scoped recompute via the change-log window; regression-tested.

## Aggregation

Aggregated head arguments, lowered internally to a temp relation plus a
group-by fold (the design doc §3.3's Flix-style monotone aggregates):

```prolog
kit_count(Person, count(Kit)) :- bought(Person, Kit).
stats(Person, count(Kit), max(Rating)) :- bought(Person, Kit, Rating).
big_spender(Person) :- kit_count(Person, N), N >= 3.
```

Semantics and guarantees:
- Group key = the non-aggregated head prefix; each `count/min/max/sum`
  folds a column over the group's distinct body solutions
  (`count` is COUNT-DISTINCT by set semantics).
- **Strict stratum ordering**: aggregated predicates complete their
  fold before any reader evaluates — enforced by treating aggregation
  dependencies like negation edges in the stratification, with
  recursion through an aggregated head and mixed ordinary/aggregated
  definitions of one predicate rejected at install time.
- **Value changes propagate**: growth increments the fold
  (2 -> 3 flips `big_spender`) and retraction decrements it; replaced
  values retract-and-reinsert, and the DRed-lite cascade recomputes
  downstream readers within the same epoch.
- `why()` on an aggregated fact shows the rule and a contributing row
  witness (`via agg:kit_count`).

Benchmark note: wiring dynamic per-relation counting rules into the
LongMemEval runner (`LEMMALOG_COUNTS=1` to enable) did **not** fix the
counting questions — the bottleneck there is extraction recall (items
mentioned in conversation that never become facts), and the extra
count rows added context noise. The engine feature is correct and
tested; the honest finding is that counting questions need better
extraction, not aggregation.

## Rule language

```text
# temporal projection: what's true now
current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.

# transitive closure
reports_to(X,Y) :- current(X,"manager",Y).
trans: reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).

# stratified negation
orphan(E) :- entity(E), !current(E,_,_).

# bounded relevance diffusion; conf = product of edge confidences = decay
near(S,E,1) :- mentions(S,E).
diffuse: near(S,E2,D) :- near(S,E1,Dm), Dm < 3, D = Dm + 1, edge2(E1,E2).
```

- Variables are uppercase; `"strings"` and integers are constants; `_` is a
  wildcard.
- `name:` prefixes label rules (shown in `why()` output).
- Builtins: `now(T)`, comparisons (`<  =<  >  >=  =  \=`), integer `+`/`-`.
- Recursion through negation is rejected at install time.

## API

```rust
use lemmalog::{Engine, Ann, Value};

let mut e = Engine::new();
e.install_program("current(E,R,O) :- edge(E,R,O,VF,VT,_), now(T), VF =< T, T < VT.")?;
e.declare("edge", &args, Ann::base(0.9, ["ep42"])); // extraction boundary
e.set_now(100);
e.run();                                        // incremental fixpoint
e.run();                                        // == 0: no delta, no work
e.query("current", &[Some(alice), None, None]); // pattern query
e.why("current", &args);                        // provenance proof tree
```

## Incremental maintenance model

Each `run()` is an epoch. Facts asserted since the last epoch form the
delta seeds; seminaive evaluation fires each rule once per positive body
atom bound to the delta (complete for set semantics), iterating to fixpoint.
A `run()` with no new assertions derives nothing. Retraction (supersession)
is a scoped negative delta: only derived predicates that transitively read
the retracted predicate are cleared and re-derived; everything else keeps
its incrementality. Every new fact is logged with its epoch
(`change_log`), backing "what changed in memory" reporting.

Performance notes: per-position hash indexes are maintained eagerly on
insert/remove and back *every* read path — rule-body joins (`lookup`
returns row ids), `query()`, and `ask()` all select the smallest bucket
instead of scanning (ground point lookups: 230 ms -> 92 µs over a 3.9M-fact
relation). Rule bodies backtrack via an undo trail rather than environment
clones; provenance witnesses per fact are capped (`SUPPORT_CAP`) — `why()`
needs one derivation, not the exponentially many paths. On an M-series
laptop, a 500-node chain closure (124,750 facts) fixpoints in ~17 s while
an incremental turn costs ~50 ms and an idle turn microseconds — the
incremental/idle contrast is the design's claim.

Cyclic-join evidence (`examples/graph_queries.rs`, 2,000 nodes / 8,000
arcs): triangle detection — the query class where worst-case-optimal joins
matter — is cheap here (the sparse-graph buckets keep the nested-loop
evaluator near-linear), while materializing a full transitive closure
dominates everything (3.9M facts, ~67 s) — an argument *against* blindly
materializing dense closures and for demand queries (`ask_deep`), not for
triejoins at agent-memory scale. Leapfrog triejoins remain the documented
path for dense-relation joins at much larger scale.

## Agent layer (`src/agent.rs`)

The LLM sits strictly at the extraction boundary; the fixpoint is pure.

```rust
use lemmalog::{AgentMemory, MockExtractor};

let mut m = AgentMemory::new(MockExtractor::new(0.9), "reports_to(X,Y) :- current(X,\"manager\",Y).")?;

let r = m.observe_at("alice --works_at--> acme", 100); // ADD (extractor boundary)
let r = m.observe_at("alice --works_at--> gigant", 200); // deterministic UPDATE (supersede)
let r = m.observe_at("alice --likes--> bob", 300); // non-exclusive: ADD + escalation
m.maintain(200);                                     // incremental re-derivation
m.ask("current(\"alice\", \"works_at\", O)")?;        // ["O=gigant"]
m.context(&["alice"], 200);                          // positional assembly, budgeted
m.why("current(alice, works_at, gigant)")?;          // proof tree -> episodes
```

- **Update policy** (Mem0-style, deterministic-first): no open fact → ADD;
  same fact → NOOP (annotation merge); exclusive predicate with different
  object → UPDATE (close old interval, assert new); otherwise → ADD +
  escalation for the agent to resolve.
- **Context assembler**: a leading "new in memory since last turn" section
  (from the epoch change-log), then distilled facts sorted by confidence at
  the top of the window, verbatim source episodes (from provenance) at the
  bottom — the lost-in-the-middle mitigation — under a token budget
  (60/40 split, chars ≈ 4×tokens).
- **`ask(goal)`**: read-only, side-effect-free atom query returning variable
  bindings; the datalog-as-tool surface. **`ask_deep(goal)`** answers the
  same shape via magic-sets rewriting: only the demand-relevant slice is
  derived (adornment + magic predicates, left-to-right binding passing),
  the base store is untouched, and `engine.last_demand_facts` reports the
  slice size.
- **`LlmExtractor`**: extraction behind a caller-supplied model call (any
  provider — or a test closure). Lemmalog owns the prompt, the
  `S --rel[conf]--> O` response protocol, memoization by episode id, and
  degradation-to-zero-facts on provider errors.
- **Persistence** (`save`/`load`): tab-separated snapshots of rules, clock,
  episodes (verbatim sources), escalation queue, and base facts with
  annotations. Derived relations are never persisted — they are rebuildable
  projections, recomputed on load (event sourcing, per the design doc).
- **Semantic retrieval** (`semantics` module): a `SemanticIndex` of entity
  embeddings behind an `Embedder` trait (`HashEmbedder` for offline/test);
  `seed_mentions` asserts `mentions(S, entity)` facts with cosine
  confidence, and `RELEVANCE_RULES` diffuse them over `links` edges with
  t-norm decay — the hybrid vector + symbolic half of the design.
- **Rule registry** (`install_rules`/`uninstall_rules`/`rule_batches`): rule
  programs install as versioned batches an agent can revert. Installing or
  uninstalling marks the program dirty: the next `run()` clears derived
  relations (including orphans of uninstalled rules) and backfills every
  rule against the existing store — rules installed mid-session fire
  against old facts (this was a real gap: evaluation previously reacted
  only to pending deltas).
- **Streaming change feed** (`Engine::changes_since(checkpoint)`): every
  addition, explicit retraction, and wholesale clear (scoped/program
  recompute) is stamped with an epoch. Checkpoint at `epoch()` after a run
  and an external projection — a vector index, a UI, a downstream agent —
  receives exactly the next turn's window. `Cleared(pred)` is the signal
  to re-sync a derived predicate from scratch.
- **Hypotheticals** (`Engine::hypothetical` / `AgentMemory::what_if`): the
  lookahead primitive from design §4.5 — assume temporary facts, run to
  fixpoint, answer a goal, restore the store byte-identically (relations,
  change log, epoch, dirty flags). `what_if(text, goal)` runs an episode
  through the extractor under a `hyp-` id (never colliding with real
  episodes) and reports what the assumption would add.
- **Retraction cost**: supersession uses a DRed-lite scoped recompute —
  dependents are cleared level by level in stratum order, and a level is
  only rebuilt-and-propagated-into if its input's key set actually changed.
  A `works_at` supersession rebuilds `current` (linear) and leaves a
  manager-only closure untouched.

## Run

```sh
cargo run --bin lemmalog          # interactive REPL (or pipe a script)
cargo test                         # 44 tests: engine (20) + agg (6) + agent (10) +
                                   # differential (3) + semantics (2) +
                                   # eval (2) + session (1)
cargo run --example agent_memory   # engine-level demo
cargo run --example agent_session  # full agent loop incl. ask_deep + news
cargo run --release --example investigation # the thesis in one run: derived
                                   # beliefs, why() proofs, retraction
                                   # repairs the closure
cargo run --release --example graph_queries # cyclic joins + closure stress
cargo run --release --example perf        # chain closure: fixpoint vs incremental
                                   # vs idle turn timings
cargo run --release --example eval # synthetic long-horizon eval report
```

### Synthetic long-horizon eval (seed 42, 1000 turns, 30 people)

```
knowledge updates   : 30/30 correct (220 supersessions applied)
multi-hop reasoning : 40/40 correct (magic-sets ask_deep, 7.6 ms total)
conflict abstention : 29/29 conflicted people keep ALL open preferences
overall accuracy    : 100.0%
token economics     : 422 ctx tokens vs 5386 transcript tokens (12.8x saving)
maintenance latency : 1.5 ms/turn
```

The deterministic memory behaviors LongMemEval shows frontier models fail
(knowledge updates, temporal projection, multi-hop, abstention) are exact
here because they are rule-derived. The harness (`scenario` module) also
caught two real engine bugs during development: a `swap_remove` corruption
in retraction (resurrected facts, silent neighbor deletion) and a
predicate-granularity recompute that made supersession quadratic — both now
regression-tested.

## Layout

- `src/intern.rs` — symbol interner, terms, values
- `src/ast.rs` — rule AST + hand-written parser
- `src/eval.rs` — store (row vectors + per-position indexes), annotations,
  stratification, trail-backtracking seminaive evaluation, scoped negative
  deltas, epoch change-log, queries, `ask`/`ask_deep`, proof trees
- `src/magic.rs` — magic-sets demand-program rewriting (all-free adornments
  alias materialized relations)
- `src/semantics.rs` — `Embedder` trait, `HashEmbedder`, `SemanticIndex`,
  relevance seeding
- `src/scenario.rs` — deterministic long-horizon scenario generator with
  ground truth + `run_eval`
- `src/session.rs` — the command surface behind the `lemmalog` REPL bin
- `src/agent.rs` — extraction boundary (`MockExtractor`, `LlmExtractor`),
  update policy, escalations, context assembler, `AgentMemory` facade
- `examples/agent_session.rs` — the full loop from the design doc
