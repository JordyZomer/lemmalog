# brainlog

**An agent's memory should be a deductive database.**

This repo contains **lemmalog** — a Datalog engine built as working memory
for LLM agents — plus the design document and task-prompt patterns that
motivate it.

| Path | What it is |
|---|---|
| [`lemmalog/`](lemmalog/) | The Rust crate: engine, agent layer, MCP server, REPL, skill, benchmarks |
| [`datalog-context-engine-design.md`](datalog-context-engine-design.md) | The original design document, with an honest status log of what shipped |
| [`zeroday-prompt-with-mcp.md`](zeroday-prompt-with-mcp.md) | A worked example: a multi-agent security-audit prompt that uses the lemmalog skill + MCP as its shared brain |

## The idea

LLM agents fail long horizons because context is treated as a buffer, not a
database. Lemmalog inverts that: the model asserts verified claims as
Datalog facts (with confidence and provenance); rules derive closures,
temporal views, canonicalizations and aggregations deterministically; every
derived fact answers "why do I believe this?" back to its source episodes;
and each turn updates derived views incrementally instead of re-reasoning
them in-context.

The division of labor: **the engine owns state and consequence; the model
owns perception and choice; every choice's outcome returns to the engine.**

## Quick start

```sh
cd lemmalog
cargo test                                  # 55 tests (64 with --features llm)
cargo run --bin lemmalog                    # interactive REPL
cargo run --release --example agent_session # agent loop demo
```

For the MCP server (Claude Code / Kimi CLI), the agent skill, and the
LongMemEval benchmark harness, see [`lemmalog/README.md`](lemmalog/README.md).

## License

MIT — see [LICENSE](LICENSE).
