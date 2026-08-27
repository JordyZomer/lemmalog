# Lemmalog: A Datalog Engine for LLM Context Management

**Design document — August 2026**

## 1. Problem statement

LLM agents fail on long horizons for a structural reason: *context is treated as a buffer, not a database*. The evidence:

- **Context rot** (Liu et al., arXiv:2307.03172; Chroma's 2025 context-rot study; NVIDIA RULER, arXiv:2404.06654): every frontier model degrades non-uniformly as input grows; effective reliable context is often 4k–32k tokens regardless of the advertised window.
- **Knowledge updates and temporal reasoning are the worst-performing memory abilities** (LongMemEval, arXiv:2410.10813 — frontier models drop 21–30%), because most memory systems have no principled supersession model.
- Current agent-memory systems (Zep/Graphiti, Mem0, GraphRAG, Letta) store *extracted facts* in a property graph, but **derive nothing**: closure, inheritance, contradiction detection, and consequence propagation are either re-done by the LLM on every query (expensive, unreliable) or absent.

The claim, stated narrowly: **Lemmalog treats agent memory as an incrementally maintained deductive database, combining temporal supersession, provenance-carrying derived facts, runtime-installed rules, and demand evaluation behind an agent-facing memory interface.** Adjacent work exists and predates this writing: Synalog (SynaLinks) gives agents a Datalog-family semantic layer with dynamically created entities, rules, and auditable derivations; FluctlightDB argues for a dedicated agent-memory data model from the cue/activation direction; the ProsusAI MemEval suite standardizes comparison across nine memory systems. What distinguishes this project is the combination — bi-temporal supersession plus provenance semiring plus incremental (and retraction-safe) maintenance in one engine, validated by differential testing — not the idea that logic programming can serve agents. RelationalAI demonstrated that Datalog can be a full knowledge-graph management system (Rel, arXiv:2504.10323); this project applies that machinery to the LLM context boundary, where the benchmarks that reward exactly it (temporal reasoning, knowledge updates, multi-hop) are the ones retrieval-centric systems fail.

### Why Datalog specifically

1. **Recursion is the natural shape of context queries.** "Is this claim transitively supported?", "what depends on this decision?", "which sources influenced this summary?" are reachability/dataflow queries — 2–3 Datalog rules, awkward in Cypher/SQL (SparqLog, VLDB 2023, showed Datalog is the right IR even under a SPARQL surface).
2. **Guaranteed termination and no side effects** make Datalog safe to expose to an LLM as a *query language* — the agentic-retrieval trend (text-to-SQL/Cypher) gets a strictly more declarative, safer target.
3. **Incremental evaluation is a solved problem** (differential dataflow, DDlog, DBSP/Feldera): each conversation turn is a small delta; derived views update in milliseconds instead of recomputing.
4. **Semiring annotations** (Green et al., PODS 2007; Scallop, arXiv:2302.03965) give one mechanism for provenance, confidence, and recency — the three things a memory needs — without leaving the engine.

## 2. Research foundations (what we borrow, from where)

| Lesson | Source |
|---|---|
| Seminaive evaluation over delta relations | Soufflé, Crepe, Flix, DDlog |
| Leapfrog triejoins on sorted relations (best throughput-per-complexity for an embedded Rust engine) | Datafrog (rust-lang/datafrog, Polonius lineage) |
| Lattices/semirings as first-class predicate annotations | Flix ("Fixpoints for the Masses"), Ascent, Scallop |
| Fully incremental evaluation over change streams; retractions symmetric with insertions | Differential dataflow (Materialize), DBSP (Feldera) |
| Runtime-loadable rules (an *interpreter*, not a proc-macro compiler) | Crepe/Ascent's compile-time limitation; ZodiacEdge (incremental rule-set updates, arXiv:2312.14530) |
| Magic-sets / demand-driven evaluation for point queries into large graphs | BigDatalog lineage; critical because context-selection queries touch a tiny slice |
| Bi-temporal facts, edge invalidation not deletion | Zep/Graphiti (arXiv:2501.13956), Snodgrass |
| LLM at the extraction boundary only; deterministic logic for update decisions | Mem0's ADD/UPDATE/DELETE/NOOP; Zep invalidation |
| PPR-style associative diffusion as bounded recursive rules | HippoRAG (arXiv:2405.14831) |
| Positional-bias-aware context assembly (distilled facts first, provenance verbatim last, small total) | Liu et al.; Anthropic context engineering; Zep's 90% latency / Mem0's 91% token reductions |
| Background/deferred maintenance off the hot path | Letta sleep-time compute (arXiv:2504.13171) |
| Provenance proof-trees for auditability | Green et al.; Zhao (Sydney thesis) |

**Key design constraint discovered by research:** no established system puts an LLM call *inside* a Datalog fixpoint, and for good reason — LLM calls are non-monotone and expensive. The design below keeps LLM predicates outside the fixpoint via strict stratification and memoization.

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Agent / LLM loop                       │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────────┐  │
│  │ context      │◄──│ query tool    │◄──│ memory-edit    │  │
│  │ assembler    │   │ (datalog-as-  │   │ tools (write   │  │
│  │ (positional) │   │  query)       │   │  facts/rules)  │  │
│  └──────┬───────┘   └───────┬───────┘   └───────┬────────┘  │
└─────────┼───────────────────┼───────────────────┼───────────┘
          │ assembled ctx     │ eval requests     │ deltas
┌─────────▼───────────────────▼───────────────────▼───────────┐
│                     LEMMALOG ENGINE (Rust)                  │
│                                                              │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │ Store:     │  │ Evaluator:       │  │ Rule registry:  │ │
│  │ bi-temporal│  │ seminaive +      │  │ runtime-loaded  │ │
│  │ relations, │  │ leapfrog joins,  │  │ stratified      │ │
│  │ annotated  │  │ stratified,      │  │ programs,      │ │
│  │ (semiring) │  │ demand-driven    │  │ versioned      │ │
│  └─────┬──────┘  │ (magic sets)     │  └────────────────┘ │
│        │         └────────┬─────────┘                      │
│        └──Δ change stream─┴──► incremental maintenance     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Derived views: CurrentFacts, Relevance, Contradictions│ │
│  │ Supports, Salience, CompactionInputs                  │ │
│  └───────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
              ┌────────────┴─────────────┐
              │ Ingestion stratum (LLM, │
              │ out-of-fixpoint, memoized):│
              │ OpenIE → candidate facts │
              │ contradiction ID         │
              └──────────────────────────┘
```

### 3.1 Data model: annotated bi-temporal relations

Every base relation ("EDB") tuple carries:

```prolog
% entity, relation, object, valid_from, valid_to, asserted_at, confidence, provenance
edge("alice", "works_at", "acme", 2026-01-01, ∞, 2026-08-25T10:00, 0.95, ep:1042)
```

- **Bi-temporality** (valid time vs. asserted/transaction time): updates are retraction-by-annotation (`valid_to` set), never deletion — full history preserved, "as-of" queries possible.
- **Annotations live in a semiring/lattice**, not ad-hoc columns:
  - `confidence ∈ [0,1]` with a chosen t-norm fusion (Scallop-style);
  - `provenance ∈ set of episode IDs` (set-union semiring) — every derived fact answers "why do I believe this?";
  - `salience ∈ interval lattice` for recency/frequency fusion.
- Entities are interned to integer IDs; text payloads and embeddings live in side-tables joined by ID (the engine stores relations, not blobs).

### 3.2 Derived views: the rules ARE the memory

Stratified program, versioned and hot-loadable:

```prolog
% Stratum 0 — temporal projection (what's true NOW)
now(E,R,O,C,P) :- edge(E,R,O,VF,VT,TS,C,P),
                  VF <= now(), now() < VT.

% Stratum 1 — closure & inheritance
reports_to(X,Y) :- now(X,"manager",Y,_).
reports_to(X,Z) :- reports_to(X,Y), reports_to(Y,Z).

project_member(P,Pr) :- now(P,"works_on",Pr,_).
project_member(P,Pr) :- project_member(Q,Pr), reports_to(P,Q).

% Stratum 2 — supersession / contradiction candidates
superseded(E,R,O1,O2) :- edge(E,R,O1,_,VT1,TS1,_,_),
                         edge(E,R,O2,_,_,TS2,_,_),
                         TS2 > TS1, mutually_exclusive(R).
% mutually_exclusive/1 is a small curated table; the *detection*
% of contradiction is done by rules, the *resolution* policy by rules
% or escalated to the agent.

% Stratum 3 — support propagation (why-trust)
supports(S,F) :- cites(S,F).
supports(S,F) :- supports(S,G), supports(G,F).

% Stratum 4 — bounded PPR-style associative salience
near(S,Entity,1,C) :- mentions(S,Entity,C).
near(S,E2,D+1,C') :- near(S,E1,D,C), now(E1,R,E2,Ce,_),
                     D < 3, C' = C * Ce * decay.
```

This is the core thesis: **memory = base facts + this rule layer, incrementally maintained.** Closure, supersession, support propagation, and salience diffusion are derived in milliseconds per turn instead of being re-reasoned by the LLM per query.

### 3.3 Evaluator

- **Rust interpreter** (rules runtime-loadable — Crepe/Ascent's compile-time binding rules them out; ZodiacEdge demonstrates incremental rule-set updates).
- **Seminaive evaluation** with per-atom delta rule versions; strata evaluated bottom-up in dependency order; stratified negation and lattice aggregation (Flix-style monotone aggregates) supported.
- **Leapfrog triejoins** (Datafrog lineage) over sorted tries as the join core.
- **Incremental maintenance:** change stream keyed by epoch; retractions flow symmetrically with insertions (DBSP-style integration over the change stream). Each conversation turn is a small delta; derived views update incrementally. Snapshots allow cheap rollback/branching of hypotheticals ("what if we assume X?" → evaluate against a delta-epoch without mutating the store).
- **Demand-driven mode:** magic-sets rewriting so context-selection queries ("what's relevant to entity E?") touch only the relevant slice — the common case for a context engine with a large store.
- **Complexity budget:** per-query cost estimator; recursive rules carry declared depth/height bounds (e.g., `near` above) so evaluation is trivially bounded. Non-expert users (the LLM writing queries) can't construct non-terminating programs.

### 3.4 LLM integration: strictly at the boundary

The fixpoint never contains an LLM call. The neuro-symbolic loop is:

1. **Ingest** (post-turn, async): an extraction LLM turns new episodes into candidate fact tuples (OpenIE à la Zep/Mem0) with confidence and provenance pointers. Results are **memoized** by (episode-hash, extractor-version) — never re-extracted.
2. **Update decision** (deterministic first): rule layer flags contradictions and near-duplicates. Unambiguous cases (superseded facts get `valid_to` set) are resolved by rules; ambiguous cases escalate as a small work item to the agent ("you previously believed X; new evidence says Y — resolve?"). This is Mem0's ADD/UPDATE/DELETE/NOOP pattern with rules replacing the second LLM call wherever possible.
3. **Derive** (async, incremental): the rule layer re-converges on the delta — the "sleep-time compute" slot; heavy reorganization (re-clustering, compaction) happens here, never on the hot path.
4. **Retrieve & assemble**: query time, the engine answers the relevance view; the **context assembler** places distilled high-confidence derived facts at the *beginning* of the context window, verbatim provenance excerpts at the *end* (respecting lost-in-the-middle), and keeps total injection small (the 90% token/latency reductions Zep and Mem0 report come precisely from small structured context beating big raw context).

### 3.5 Agent-facing interface

Three tool surfaces:

- **`lemmalog.query(datalog)`** — the agent writes bounded Datalog against the views (safe: declarative, side-effect-free, guaranteed to terminate with the depth bounds). Datalog-as-query-interface is an open niche vs. text-to-SQL/Cypher; its guaranteed safety is the selling point.
- **`lemmalog.declare(facts)` / `lemmalog.install(rules)`** — agent-writable structural memory (Letta's insight: *retrieval is not memory*; the agent must be able to *write state*, here as facts and versioned rules).
- **`lemmalog.why(fact)`** — returns the provenance proof tree (which episodes, which rules, which confidence fusions produced this fact). The audit primitive that no vector store can offer, and the debugging interface for the memory itself.

### 3.6 Storage & persistence

- Append-only fact log (event sourcing; the episode log is the source of truth, the semantic graph is a rebuildable projection — Zep's architecture).
- Columnar relation snapshots + WAL; epochs align with conversation turns for point-in-time replay ("what did the agent believe at turn 40?").
- Embeddings as a side index keyed by entity ID, seeding the `near` diffusion rule — hybrid vector + symbolic, per the 2025–26 consensus.

## 4. Why this improves LLM reasoning (mechanism, not hope)

1. **Offloads exactly what LLMs are worst at.** Multi-hop transitivity, supersession, and temporal projection are solved *exactly* by rules — and are the lowest-scoring abilities in LongMemEval. The LLM spends its window on generation and judgment, not bookkeeping.
2. **Small, positioned, structured context beats large raw context** — established empirically (context rot; Zep/Mem0 latency and token results). Derived views are the compaction strategy with *definitions*: a summary that can answer "why."
3. **Provenance kills hallucinated memory.** Every injected fact carries a proof tree back to source episodes; the model can be *prompted with the proof*, anchoring generation (FiDeLiS, ICLR 2025, showed KG-anchored reasoning paths improve faithfulness).
4. **Deterministic knowledge updates.** Contradiction handling becomes a data problem with a policy, not a per-query LLM judgment — the single largest measured failure mode of long-horizon agents.
5. **Cheap hypotheticals.** Delta-epoch evaluation gives the agent "what follows if we assume X?" in milliseconds — a primitive for lookahead/planning that pure-LLM reasoning cannot do reliably.

## 5. Implementation plan

> **Status (2026-08):** Phases 1–5 are implemented in [`lemmalog/`](lemmalog/) — runtime-parsed stratified Datalog with negation, semiring annotations (confidence × provenance), seminaive incremental maintenance, DRed-lite scoped negative deltas, `why()` proof trees, the agent layer (pluggable `LlmExtractor`, deterministic update policy with escalations, positional context assembler with a "new in memory" epoch feed, `ask()` + magic-sets `ask_deep()`), per-position join indexes with trail backtracking, snapshot persistence (event-sourced), a semantic side index (`Embedder` trait + relevance diffusion), lattice-style aggregation (count/min/max/sum head arguments with group-by folds, strict stratum ordering, and value-change propagation — with the honest benchmark finding that LongMemEval counting questions are bounded by extraction recall, not aggregation), a versioned rule registry (agent-installable/revertable, with backfill on change), hypothetical `what_if` lookahead (design §4.5), and a Phase-5 synthetic eval harness, differential correctness testing against a naive fixpoint oracle (which caught and fixed a cross-run negation soundness bug), an interactive REPL, live real-model integration against both LM Studio and hosted Claude (Opus 4.8 authors correct recursive rules in seconds — validated, installed, backfilled, ground-truth-verified, reverted — and scores 3/4 on the memory battery in 14s) (qwen3.8-27b + nomic-embed behind `--features llm`: schema-constrained extraction, strict protocol validation, speaker resolution; live runs show deterministic supersession/temporal/abstention exact, with residual failures traced to the model side) — 42 passing tests, 100% accuracy on the deterministic suite at 1.5 ms/turn maintenance and 12.8× token savings (1000-turn scenario). LongMemEval (oracle) complete 30-instance run with Opus: overall F1 0.48 vs 0.50 for the transcript baseline (statistical tie) with 12/30 vs 10/30 exact matches and 1.4-30x smaller contexts — same-or-better accuracy at an order of magnitude less context; single-session-user 5/5 exact for memory including one baseline miss; counting questions expose the missing aggregation feature. Remaining: leapfrog triejoins (benchmarked as unnecessary at agent scale — see `graph_queries`), real-benchmark (LongMemEval) integration; the streaming-delta story is covered by the epoch change feed (`Added`/`Retracted`/`Cleared`) plus DRed-lite scoped recomputes. See `lemmalog/README.md`.

**Phase 1 — Core (Rust crate `lemmalog-core`)**: typed relations, interned entities, bi-temporal annotations, seminaive interpreter with stratified negation, leapfrog joins, unit-tested against Soufflé outputs on the Datalog benchmarks. ~The Flan benchmark suite gives the reference performance targets.

**Phase 2 — Incrementality**: change streams, epochs, delta maintenance (DBSP-inspired), snapshots, magic-sets demand mode. Benchmark: per-turn re-derivation under 10 ms for stores of ~10M tuples on typical agent-scale deltas.

**Phase 3 — Semirings & provenance**: annotation polymorphism (boolean → confidence t-norm → provenance set), `why/1`, proof-tree rendering into model-consumable text.

**Phase 4 — LLM layer**: extraction pipeline with memoization; contradiction-resolution rules; context assembler with positional placement policy; tool surfaces (`query`, `declare`, `install`, `why`).

**Phase 5 — Eval**: LongMemEval (the honest benchmark — knowledge updates, temporal reasoning, abstention; treat LoCoMo leaderboard claims skeptically, judge-model choice swings scores by tens of points). Ablate each lever (temporality, derived closure, provenance injection, positional assembly) to attribute gains. Also measure: token cost per session vs. full-context baseline, per-turn latency.

**Risks:**
- *Extraction quality bounds system quality* — garbage rules over garbage facts. Mitigation: confidence annotations end-to-end; agent-facing resolution queue.
- *Rule-base drift*: agent-installed rules may conflict. Mitigation: versioned, stratification-checked rule registry; rules carry provenance too and are revertable.
- *Semiring generality vs. speed*: fix one concrete annotation product (confidence × provenance × validity) first; generalize after the eval harness exists.

## 6. Positioning

| System | Temporal | Derived rules | Incremental | Provenance | LLM-safe query lang |
|---|---|---|---|---|---|
| Zep/Graphiti | ✔ (bi-temporal) | ✘ | partial | episodes only | ✘ (Cypher-internal) |
| Mem0 | partial | ✘ | ✘ | ✘ | ✘ |
| GraphRAG/LightRAG | ✘ | ✘ | partial (LightRAG) | ✘ | ✘ |
| Letta blocks | ✘ | ✘ | n/a | ✘ | ✘ |
| **Lemmalog** | ✔ bi-temporal | ✔ stratified datalog | ✔ delta-based | ✔ proof trees | ✔ bounded datalog |

The one-sentence pitch: **Zep proved agents need a temporal knowledge graph; RelationalAI proved Datalog is a knowledge-graph engine; Lemmalog combines them — the agent's memory becomes a deductive database, and the LLM stops paying tokens to re-derive what rules derive in microseconds.**

---

### Key references

- Engines/semantics: Soufflé (CAV'16) · Flix (arXiv:2304.03634) · Datafrog (rust-lang) · Crepe/Ascent · DDlog · DBSP (SIGMOD'23) · Rel (arXiv:2504.10323) · Scallop (arXiv:2302.03965) · Green et al. provenance semirings (PODS'07) · SparqLog (VLDB'23) · ZodiacEdge (arXiv:2312.14530)
- LLM memory/context: Liu et al. lost-in-the-middle (arXiv:2307.03172) · Chroma context rot · RULER (arXiv:2404.06654) · Anthropic context engineering · Zep/Graphiti (arXiv:2501.13956) · Mem0 (arXiv:2504.19413) · HippoRAG (arXiv:2405.14831) · A-Mem (arXiv:2502.12110) · MemoRAG (arXiv:2409.05591) · Cognee (arXiv:2505.24478) · Letta sleep-time compute (arXiv:2504.13171) · LongMemEval (arXiv:2410.10813) · Logic-LM (arXiv:2305.12295) · FiDeLiS (ICLR'25)
