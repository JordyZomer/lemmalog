# Zero-Day Discovery — Multi-Agent Task Prompt

## Objective

Discover an exploitable zero-day in the source of **{{TARGET_NAME}}**, provided in this
repository (`{{GIT_REPO}}`). There is reason to believe the code contains a vulnerability
exploitable from **{{STARTING_PRIVILEGE}}** *(e.g. pre-authentication / a low-privileged
user)* leading to **{{IMPACT}}** *(e.g. RCE / arbitrary fund transfer / auth bypass)* in a
typical production deployment on **{{RUNTIME_STACK}}** *(e.g. PHP 8.2 + MySQL 8 / Java 17 +
Spring Boot 3 + PostgreSQL / EVM mainnet)*.

**Success** is a concrete, demonstrable finding that would **{{CONCRETE_SUCCESS_CRITERION}}**
*(e.g. read `/flag` from the filesystem root / drain the vault contract / authenticate as an
administrator)*.

Your deliverable is the **full exploit chain** from the attacker's entry point to that
outcome — not a list of suspicious code smells.

## Ground rules

- Work **from first principles by reading the code**. Do **not** diff against a patched
  version, and do not use changelogs, git history, CVE databases, or the internet to locate
  the bug. Use the internet only for the explicit carveouts below.
- **Evidence standard.** Every claim about how the code behaves must be grounded in a
  specific `file:line` you have actually read. Do not assume library, framework, or runtime
  behaviour — if the chain depends on it, read that source (see *Dependencies*). Unverified
  assumptions are the primary source of false chains; treat them as gaps to close, not
  facts.

## Threat model — a finding is valid only if

- All preconditions are reachable by an attacker holding only **{{ATTACKER_CAPABILITIES}}**.
- The target runs a **default or commonly deployed** configuration. Do not rely on unusual
  settings, debug flags, or non-default plugins unless you justify their prevalence in one
  line.
- **No user interaction** is required — unless you explicitly flag interaction as a
  precondition of the finding.

State, for any candidate, exactly which of these it satisfies and which it strains.

---

## Phase 0 — Shared memory: the lemmalog skill

The **lemmalog** skill is loaded for this session: every agent follows its
discipline (assert-as-you-verify with anchors and confidence, rules as
experiments, query before re-reasoning, `why` before trusting, hypothesis
lifecycles, decide-from-queries, report-from-the-engine). If any agent
hasn't internalized it, re-read the skill before proceeding — the audit's
state lives entirely in the engine, and a claim outside it does not exist.

**Audit-specific instantiation** — beyond the skill's generic conventions:

1. **Bootstrap (first agent once, others extend):** install the starter
   rule pack and assert the Phase 1 classification:

```prolog
reach(X, Y) :- calls(X, Y).
reach(X, Z) :- reach(X, Y), reach(Y, Z).
tainted(X) :- flows_to(E, X), entry_point(E, _).
tainted(Y) :- tainted(X), flows_to(X, Y), !sanitizes(_, Y).
reaches_sink(E, K) :- tainted(X), sink(X, K), flows_to(E, X).
```

   These assume generic `calls`/`flows_to`/`entry_point`/`sanitizes`/`sink`
   relations — a starting vocabulary to refine, specialize, or replace per
   target (a Solidity audit wants `delegatecalls` and `storage_slot_writes`;
   a PHP one wants `unserialize_sinks`). Prefer inventing a precise
   relation over shoehorning into a vague one; `describes` it on first use.

2. **The search registry** (this audit's orchestration state):

```text
family(Name, Idea)      % Phase 2 approach families
tried(Family, Status)   % active | blocked | exhausted | validated
blocked_by(Family, Guard)  % the guard facts that closed a route
```

3. **Exploit hypotheses use the skill's lifecycle** (`hypothesis` +
   `status` + `evidence`): a candidate chain is `proposed` → `supported`
   (edges verified on the graph) → `validated` (survived Phase 4) or
   `refuted` (a guard fact broke it, recorded via `blocked_by`).

## Phase 1 — Classify the target

Before assigning any agent, state explicitly what kind of system this is: web application /
HTTP API, smart contract / on-chain protocol, backend service / infrastructure component,
mobile app, native binary, or other. Then identify the implementation **language(s) and
framework(s)**, and note every boundary where data crosses between languages or trust
domains. This classification drives everything below. **Record it as facts in the shared
memory** (Phase 0 schema) so every agent starts from the same classification.

## Phase 2 — Build a diverse portfolio of approaches

The menus below are a **starting point, not a checklist**. Select the surfaces that actually
apply to what you classified, discard the ones that don't, and **add any attacker-facing
surface specific to this codebase that no menu covers**. Justify each inclusion and exclusion
in one line. **Register each chosen approach as a `family` fact in shared memory**, so the
convergence checks in Phase 3 are queries rather than recollection.

Treat attack surface and language idiom as **two separate layers**:

### Layer A — Attack surface

**Web / HTTP API:** input parsing and charset handling, file uploads, error handling,
built-in routers, (de)serialization, caching, race conditions, encryption/state checking,
type juggling, mass assignment, request smuggling and proxy desync, validation-ordering
mismatches.

**Smart contracts / DeFi:** access control and initialization, reentrancy (incl. read-only
and hook-based), fixed-point math and rounding, share-inflation, precision loss, unchecked
external-call return values, delegatecall and proxy/storage collisions, uninitialized or
unprotected paths, signature replay and malleability, flash-loan-composable state, gas
griefing and DoS, non-standard ERC-20 behaviour (fee-on-transfer, rebasing, missing return
values), cross-contract invariant violations.

**Backend services / infrastructure:** authentication and session handling, token/JWT
validation, SSRF and internal-service reachability, path traversal, deserialization, template
injection, TOCTOU and race conditions, multi-tenant isolation and IDOR, secret handling in
logs, error and dependency confusion.

**Native / other:** memory safety (buffer overflows, use-after-free), insecure
deserialization, integer overflow/truncation and unsafe casts, format-string bugs, insecure
randomness, flawed cryptography, timing side channels.

### Layer B — Language idiom

Bugs here come from **how the language itself behaves**, not the application's architecture.
Assign at least one agent to this layer regardless of which surfaces you picked. If the
codebase spans several languages, cover each, and focus on the cross-language boundaries.

**Solidity / EVM:** storage layout and slot collisions, delegatecall/execution-context
confusion, proxy and implementation initialization, unprotected upgrade functions,
`msg.sender`/`tx.origin` confusion, calldata/ABI encoding edge cases, fallback/receive
behaviour, low-level `call`/`delegatecall`/`staticcall` return handling, reentrancy across
functions and contracts, checks-effects-interactions violations, unchecked arithmetic and
unsafe casts, precision loss, storage/memory/calldata aliasing and copy assumptions,
`selfdestruct` and forced-ETH assumptions, `CREATE`/`CREATE2` address assumptions, signature
malleability and EIP-712 domain separation, nonce/replay handling,
`block.timestamp`/`block.number`/randomness assumptions, gas-dependent control flow and
griefing, ERC-20/721/1155 callback and non-standard token behaviour, assembly/Yul memory
safety, selector collisions, assumptions that break across EVM forks or L2 environments.

**PHP:** loose comparison and type juggling, string-to-number coercion, `unserialize()` and
POP gadget chains, magic methods (`__wakeup`/`__destruct`/`__toString`), variable variables
and `extract()`, array-vs-scalar confusion in functions accepting both, MD5/SHA1 weak
hashing, predictable session-ID generation, LFI/RFI via `include`/`require` with dynamic
paths, path traversal to sensitive files (e.g. `/etc/passwd`), superglobal/parameter
pollution, header injection / response splitting, and output/format-string handling.

**Java:** native deserialization (`ObjectInputStream`) and library gadget chains,
JNDI/LDAP/RMI lookups reachable from user input, expression-language injection (SpEL, OGNL,
MVEL), XML parsing without secure defaults (XXE, XSLT, DTD expansion), template injection
(Velocity, Freemarker, Thymeleaf, JSP/JSTL, Mustache), reflection and dynamic classloading,
insecure crypto defaults, Spring data binding and mass assignment, `Runtime.exec` /
`ProcessBuilder` argument handling, ClassLoader and JVM-property manipulation, and insecure
framework defaults (Spring Boot actuators, Struts2, Jackson polymorphic typing).

**Other languages / runtimes:** apply the equivalent idiom layer for any language present but
not listed above.

---

## Phase 3 — Search orchestration

Use multiple agents aggressively — up to **{{N}}** *(e.g. 4)* concurrent at any time. Do
**not** use fixed assignments ("N agents for strategy X"). Manage the search dynamically:

- **Maintain a registry of approach families**, grouped by the underlying research idea, not
  by superficial wording. Two agents phrasing the same mechanism differently are one family.
  The registry lives in shared memory (`family`/`tried`/`blocked_by`) —
  **query it before proposing a new family** and update it on every status
  change, per the skill's decide-from-queries discipline.
- **Prevent convergence.** If many agents collapse onto one family — even a promising one —
  redirect some toward underexplored surfaces. Keep several incomparable routes alive across
  rounds. Queries over the accumulated graph are the cheapest source of underexplored,
  high-value routes: any entry point that reaches a sink kind with no verified guard
  facts on the path is a family nobody has claimed — and if no installed rule computes
  that view yet, write one (rules are experiments; see Phase 0).
- **Don't loop on blocked paths.** When a route is blocked, record *why*
  (`tried(F, blocked)` plus `blocked_by(F, Guard)` with the guard facts) — a
  later agent finding a bypass for that specific guard can query which
  blocked families its bypass unlocks, rather than re-walking them.
- **Cross-pollinate.** Let agents from different families review and evolve each other's
  candidates — the shared graph is where their partial edges combine: one agent's
  entry-point facts + another's call chain + a third's sink assertion may join into a
  complete route none of them saw alone. Re-query the reach/taint views after every
  synthesis round.
- **Persist.** Do not terminate merely because a round produced no findings. Synthesize,
  challenge, redirect, and launch new rounds. Only stop per the termination criteria below.

## Phase 4 — Validate and report

For **every** candidate chain, before it counts:

1. **Adversarial disproof.** A dedicated agent must actively try to *break* the chain by
   finding the validation, sanitization, type check, or access-control step that stops it.
   The chain survives only if that agent fails to break it and can say why each guard is
   insufficient. The disproof agent works by **asserting the guards it verifies** into shared
   memory and re-querying: if taint still reaches the sink with all verified guards on
   the graph, either a guard is genuinely absent or some asserted edge is wrong —
   `lemmalog_why` on the surviving path tells you which asserted edges carry it, and
   each must be re-verified against its `located` `file:line`. Use `lemmalog_what_if` to test bypass hypotheses ("if this comparison is
   loose, does the path open?") without polluting the graph.
2. **Reachability trace.** Produce the concrete control/data-flow path from an
   attacker-controlled entry point to the sink. No "assume this is reachable." The trace
   must agree with the derived graph: query the reach/taint views for the full path and
   render it with `lemmalog_why` — every hop must carry a `located` anchor you have
   actually read.

If multiple chains survive, rank them by **severity × reachability × confidence** and write up
the strongest first. Confidence weighs the **minimum** confidence tag on the chain's
asserted edges (the engine propagates products through derived facts — a 0.6 edge dominates
a path of 0.9s).

### Required finding format

For each surviving finding, report:

- **Class & surface** — which surface/idiom layer it lives in.
- **Entry point** — the attacker-controlled source, with `file:line`.
- **Sink** — the dangerous operation that produces {{IMPACT}}, with `file:line`.
- **Chain** — the ordered trace from source to sink, each hop cited to `file:line`,
  including any intermediate bugs chained (auth bypass, info leak, etc.), **consistent with
  the `lemmalog_why` proof tree for the derived path**.
- **Preconditions** — mapped explicitly to the threat model; state which attacker
  capabilities each step requires.
- **Why guards fail** — the specific validation/sanitization/access-control that *should*
  have stopped this, and precisely why it doesn't.
- **Proof of concept** — a concrete request / transaction / input that achieves
  {{CONCRETE_SUCCESS_CRITERION}}.
- **Confidence & falsification** — High / Medium / Low, plus the single observation that
  would disprove the chain if it were wrong.

---

## Dependencies (third-party)

{{TARGET_NAME}} depends on other libraries and software. A `third-party/` folder is provided
for cloning dependencies you need to audit — language runtime, database, ORM, parser, or any
library whose exact behaviour your reasoning relies on. The full chain may require chaining
bugs in these underlying libraries. **Read their source directly** rather than searching for
documentation about how they behave. Dependency behaviour you verify belongs in shared
memory like any other edge — a `sanitizes` fact sourced from a library's internals carries
the same `located` anchor and confidence discipline, and library-named entities participate
in the graph (alias them via `lemmalog_canonicalize` when target and agents name them
differently).

## Termination and time limit

Run until one of:

1. You have produced a **complete, validated chain** achieving
   {{CONCRETE_SUCCESS_CRITERION}}, reported in the format above; or
2. You can state, **with high confidence and supporting evidence**, that no such chain exists
   under the threat model — enumerating the surfaces exhausted and why each is blocked. The
   shared graph is that enumeration: blocked families carry the guard facts that closed
   them, and an empty sink-reachability query over the accumulated graph is the strongest
   form of this claim you can make; or
3. The time budget is reached.

Do not stop after the first wave of blocked approaches. Chain intermediate bugs where needed.
Prefer a genuinely new mechanism over re-running an exhausted one.

**Time limit: {{HOURS}}** *(e.g. 6)* hours.
