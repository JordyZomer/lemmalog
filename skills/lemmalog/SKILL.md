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
  registration; `lemmalog_save` forces a snapshot.

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
   facts plus their verbatim source episodes under a token budget — use it
   instead of `lemmalog_dump` when preparing answers; selection beats
   dumping.
4. **`lemmalog_why` before trusting any derived fact.** The proof tree
   shows which asserted edges carry it; a chain is only as good as its
   lowest-confidence edge. Re-verify the weakest edges against their
   `located` anchors.
5. **Hypotheses have lifecycles.** `hypothesis(H, "claim")` plus
   `status(H, proposed|supported|refuted|validated)` and evidence links.
   Test counterfactuals with `lemmalog_what_if` — temporary facts,
   answered goal, store untouched.
6. **Reconcile vocabulary, don't enforce it.** Name things naturally;
   when two names mean one thing, `local --alias_of[conf]--> canonical`
   via `lemmalog_canonicalize`. Conflicts surface as `alias_conflict`
   facts; they never silently merge.
7. **Decide from queries.** Derive candidate views — unexplored items,
   blocked-by-what, what-needs-attention — and choose among them. The
   queries propose; you dispose (including off-list when judgment says
   so). Then assert the decision so state stays complete.
8. **Report from the engine.** Final deliverables render from queries and
   `why` trees, not from memory. A conclusion's confidence is its minimum
   edge confidence.

## Schema conventions

The only shared vocabulary (everything else: invent precisely, and assert
`describes(Relation, "one-line meaning")` so others discover it):

```text
located(Entity, "ref")        % evidence anchor for anything in a source
describes(Relation, "what")   % self-documenting schema
hypothesis(H, "claim")        % lifecycle-tracked claims
status(H, S)                  % proposed|supported|refuted|validated
evidence(H, Fact)             % link a hypothesis to its supporting facts
decision(Id, What)            % choices made, so state stays complete
```

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

## Boundary

Stays in your head: in-flight reading, semantic judgment.
Must land in the engine: conclusions, state changes, decisions — before
you move on.
