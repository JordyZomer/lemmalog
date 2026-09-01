---
name: lemmalog
description: >-
  Externalize working memory and logical state into the lemmalog Datalog
  engine (MCP). Use for ANY multi-step task where state should outlive one
  context window or span agents: long investigations, debugging sessions,
  audits, multi-agent searches, systematic explorations, planning with many
  interdependent constraints, anything needing provenance for its
  conclusions. Trigger when lemmalog_ MCP tools are available and the task
  involves accumulating verified facts, tracking hypotheses or status over
  time, or repeatedly re-deriving the same relationships.
---

# Lemmalog — external working memory

`lemmalog` is a Datalog engine exposed as MCP tools. It is your working
memory: assertions with provenance and confidence, derived consequences
computed by rules, hypotheses with lifecycles — persistent across agents
and context resets.

**The division of labor:** the engine owns state and consequence; you own
perception and choice; every choice's outcome returns to the engine.
A claim that isn't in the engine doesn't exist — nothing durable lives in
your context.

## Setup

- The `lemmalog_*` tools must be registered (see the README of the
  lemmalog repo). If they are absent, tell the user the one-line
  registration command and continue without memory — never block the task.
- Persistence across sessions exists if `LEMMALOG_MCP_PATH` was set at
  registration; `lemmalog_save` forces a snapshot. Snapshots carry rule
  batches too — installed analyses survive restarts under their batch ids.
- No MCP access (sub-agents that don't inherit MCP connections, scripts,
  cron)? `lemmalog-cli` works on the SAME snapshot:
  `LEMMALOG_MCP_PATH=... lemmalog-cli observe --facts 'S --rel--> O'`,
  plus query/retract/context/why/rules/dump. Mutations are visible to
  the MCP server on its next load and vice versa — but the two hold
  separate in-process copies, so don't write from both at once: hand
  sub-agents the CLI and keep the parent on it too, or have the parent
  only read while a sub-agent writes.

## The discipline

1. **Assert as you verify.** The moment you confirm something — a call
   edge, a config value, a decision, a step completed — assert it:
   `S --rel[conf]--> O`. Tag read-and-verified facts `[1.0]` and inferences
   `[0.4]`–`[0.7]`. Do not omit the tag on anything you expect to derive over:
   the default is 0.9 and confidence is a product down the proof chain, so four
   hops of "verified" facts land at 0.66 and a deep closure decays to noise.
   Anchor evidence with `located(Entity, "file:line")` (or any stable
   reference) so provenance survives derivation — source references are valid
   entity tokens as long as they contain no spaces. Never assert what you
   haven't checked.
2. **Install rules when a pattern repeats.** If you ask the same shape of
   question twice, write the Datalog for it: transitive closures, guard
   tracking, status rollups, `count`/`min`/`max`/`sum` aggregates. Rules
   are experiments: one named batch per analysis idea, validated on
   install (rejections are spec feedback), backfilled against everything
   already asserted, `lemmalog_uninstall` when the idea dies.
3. **Query before re-reasoning.** Multi-hop, transitive, or
   not-X-reachable questions go through rules and `lemmalog_query` —
   never mental closure, and never re-deriving what a prior agent derived.
   For grounded answering, `lemmalog_context` retrieves the question-relevant
   facts plus their verbatim source episodes under a token budget, with an
   attribution contrast (which subjects hold facts on the topic — a
   question-mentioned party with zero topic facts is a false-premise
   signal) and, for current-state questions, the latest value per slot
   with supersessions as history — use it instead of `lemmalog_dump` when
   preparing answers; selection beats dumping.
4. **`lemmalog_why` before trusting any derived fact.** The proof tree
   shows which asserted edges carry it; a chain is only as good as its
   lowest-confidence edge. Re-verify the weakest edges against their
   `located` anchors.
5. **Hypotheses have lifecycles.** `H --hypothesis--> claim`,
   `H --status--> proposed|supported|refuted|validated` (supersedes),
   `H --evidence--> ref` (accumulates).
   Test counterfactuals with `lemmalog_what_if` — temporary facts,
   answered goal, store untouched.
6. **Correct by retracting.** When you learn an asserted fact was
   wrong — not changed, wrong — `lemmalog_retract` it: the response
   lists every derived conclusion that died with it, so the repair is
   visible, not silent. (A value that merely changed is re-asserted
   under the same relation; see "State that changes".) After a context
   reset or another agent's turn, `lemmalog_changes` with your last
   epoch resyncs you without re-reading the store.
7. **Reconcile vocabulary, don't enforce it.** Name things naturally;
   when two names mean one thing, `local --alias_of[conf]--> canonical`
   via `lemmalog_canonicalize`. Conflicts surface as `alias_conflict`
   facts; they never silently merge.
8. **Decide from queries.** Derive candidate views — unexplored items,
   blocked-by-what, what-needs-attention — and choose among them. The
   queries propose; you dispose (including off-list when judgment says
   so). Then assert the decision so state stays complete.
9. **Report from the engine.** Final deliverables render from queries and
   `why` trees, not from memory. A conclusion's confidence is the product
   of its edges (the engine multiplies down the proof chain) — deep
   derivations need high-confidence inputs to stay believable.

## Schema conventions

The only shared vocabulary (everything else: invent precisely, and assert
`describes(Relation, "one-line meaning")` so others discover it):

All asserted through the line protocol (the predicate forms below are
descriptions, not assertable syntax):

```text
kernel_func --located--> vm/vm_map.c:3052     % evidence anchor (multi-valued)
works_at --describes--> person is employed at % self-documenting schema
hyp_1 --hypothesis--> claim in plain words    % lifecycle-tracked claim
hyp_1 --status--> proposed                    % supersedes on change:
                                              % proposed|supported|refuted|validated
hyp_1 --evidence--> vm_map.c:3052            % multi-valued: accumulates
decision_7 --decision--> chose scope X because Y
```

Evidence objects take a bare source reference (space-free) or a
punctuation-free phrase; spaces plus punctuation read as leaked prose
and are dropped. Symmetric quotes around subjects/objects are stripped
(`"mean field"` lands as `mean field`) — multi-word values are fine up
to 8 words; compress longer prose into a short name or split it.

## State that changes

Values, quantities, and sets evolve — assert them so the engine can
maintain them (these conventions are what the update policy and the
aggregates need):

- **Update by re-asserting the same relation.** When a value changes,
  assert the new value under the SAME relation name: the policy
  supersedes the old fact automatically. Never invent a synonym
  relation for the new value (`uses` → `switched_to`) — that leaves
  both values open, and every current-state query gets flaky. If you
  need the history, the superseded fact is still queryable by its
  validity interval.
- **Bare numbers are integers.** `launch --monthly_cost--> 120` (never
  `$120` or `120 dollars`) — digit-only objects feed `sum`/`count`
  aggregates and `<`/`>=` comparisons. Mixed forms are opaque symbols.
- **Bare dates order correctly.** `moved_on` with `YYYY-MM-DD` (or
  `YYYY-MM`) objects; derive orderings with a rule
  (`earlier(A, B) :- on(A, D1), on(B, D2), D1 < D2`) rather than
  judging from prose.
- **Evolving sets: one fact per item, plus lifecycle verbs.** Track a
  watchlist/checklist as `added(X)` per item and `watched(X)`/`done(X)`
  when consumed; current membership is then a rule —
  `pending(X) :- added(X), !watched(X).` — not something you recount.
- **Conditional preferences stay conditional.**
  `prefers_when(user, lively, with_friends)` — never assert the
  condition itself as a fact unless the source says it holds now.

## Grammar

- Bare capitalized words are **variables** — quote entity names:
  `reports_to("Alice", Y)`, never `reports_to(Alice, Y)`.
- Fact line protocol: `S --rel[conf]--> O`, one per line.
- **An asserted fact is `current(S, rel, O)`, not `rel(S, O)`.** Rule bodies
  match the triple: `reaches(X, Y) :- current(X, depends_on, Y).` Writing
  `depends_on(X, Y)` instead installs cleanly, reports a backfill count, and
  then derives nothing — the failure is silent, so check a new rule with one
  `lemmalog_query` before building on it.
- Rule syntax: `head(X, Y) :- atom(X, Y), X \= Z.` with `!atom` negation,
  `now(T)`, comparisons, arithmetic; aggregates only in heads.
- Errors are actionable: every `isError` result carries the offending
  input, the reason, and a hint — fix and resend; `lemmalog_observe`
  reports dropped lines with reasons, so a zero-add result is loud, not
  silent.

## Anti-patterns

- Guesses as untagged facts (tag low confidence or don't assert).
- Batching assertions to the end (assert as you verify).
- Encoding your judgment as rules (queries inform; you decide).
- Trusting derived facts without `why`.
- Re-deriving in context what the engine already closes.
- Letting two names for one thing drift (alias them).
- Renaming a relation when its value changes (supersede, don't fork).
- Numbers or dates buried in prose objects ("about $50", "last March")
  — bare values are what the engine can aggregate and order.

## Boundary

Stays in your head: in-flight reading, semantic judgment.
Must land in the engine: conclusions, state changes, decisions — before
you move on — and dead ends most of all: a searched-and-ruled-out
avenue is the most valuable thing a future agent can inherit (`X
--dead_end--> why it failed, where confirmed`). Assert them in bulk —
one observe call, one line each; fifty at once is fine.

Scope honesty: for a short task that fits one context window, working
memory in your head is cheaper — lemmalog pays when state must outlive
a window, span agents, or survive a restart. A single-session audit
with four items to track is overhead; a multi-day investigation or a
swarm reading each other's dead-ends is the payoff case.
