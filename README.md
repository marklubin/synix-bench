# synix-bench

Unified benchmark platform for evaluating agent memory and context management strategies. Combines two complementary benchmark suites under one CLI:

- **SWE-bench suite**: Tests 10 context management strategies on real-world coding tasks from the [SWE-bench](https://www.swebench.com/) dataset. Agents execute inside containers, producing patches that are verified against ground-truth test suites.
- **LENS suite**: Tests 17+ memory system adapters on longitudinal episodic reasoning scenarios. Agents answer questions about streaming episode data, scored on a 9-metric composite including evidence grounding, temporal reasoning, and answer quality.

Both suites share core infrastructure: LLM clients, tool executors, budget enforcement, metering, scoring, and statistical analysis.

## Quick Start

```bash
# Install
git clone https://github.com/marklubin/synix-bench.git
cd synix-bench
uv sync

# Verify installation
synix-bench --version
synix-bench smoke --suite lens

# List what's available
synix-bench list suites
synix-bench list strategies    # SWE-bench context strategies
synix-bench list adapters      # LENS memory adapters
```

## Usage

### Running Benchmarks

```bash
# LENS: null adapter with mock LLM (no API key needed)
synix-bench run --suite lens --strategy null --provider mock

# LENS: sqlite-hybrid adapter with GPT-4o
synix-bench run --suite lens --strategy sqlite-hybrid \
  --model gpt-4o --api-key-env OPENAI_API_KEY

# SWE-bench: naive strategy on a single instance
synix-bench run --suite swebench --strategy naive \
  --instance-id django__django-16139 \
  --model Qwen/Qwen3-32B \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY

# SWE-bench: stack+heap strategy with layout config
synix-bench run --suite swebench --strategy stack_heap \
  --layout configs/layouts/order-conv-mask.json \
  --model Qwen/Qwen3-32B \
  --base-url "$ENDPOINT" \
  --api-key-env VLLM_API_KEY
```

### Sweep (Parallel Matrix Runs)

```bash
# Run multiple configs in parallel
synix-bench sweep --config configs/experiments/matrix.json --workers 6
```

Sweep config format:

```json
{
  "configs": [
    {"suite": "swebench", "strategy": "naive", "llm": {"model": "gpt-4o"}},
    {"suite": "swebench", "strategy": "window", "llm": {"model": "gpt-4o"}},
    {"suite": "lens", "strategy": "sqlite-hybrid", "llm": {"model": "gpt-4o"}}
  ]
}
```

### Scoring and Analysis

```bash
# Score LENS results
synix-bench score --results results/lens_*.json --tier 1

# Generate report
synix-bench report --results results/ --output report.html

# Compare runs
synix-bench compare --results results/run1.json results/run2.json
```

### Configuration

Runs can be configured via CLI flags or a JSON config file:

```bash
# Via config file
synix-bench run --config my_config.json

# Via CLI flags (all options)
synix-bench run --suite swebench \
  --strategy masking \
  --model gpt-4o \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --max-steps 30 \
  --timeout 1800 \
  --workers 6 \
  --budget standard \
  --sample 50 --seed 42 \
  --trials 3 \
  -v
```

See [docs/configuration.md](docs/configuration.md) for the full configuration reference.

## Architecture

```
                    ┌─────────────────────┐
                    │   CLI (Click)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     RunEngine       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼─────┐  ┌──────▼───────┐  ┌─────▼──────┐
   │ SWEBenchSuite  │  │  LENSSuite   │  │ FutureSuite│
   │ (10 strategies)│  │(17 adapters) │  │            │
   └──────┬─────────┘  └──────┬───────┘  └────────────┘
          │                    │
   ┌──────▼────────────────────▼───────┐
   │         Shared Layer              │
   │  LLMClient · ToolExecutor ·      │
   │  BudgetEnforcement · Metering ·  │
   │  Scorer · Analysis               │
   └───────────────────────────────────┘
```

The platform is built around four core abstractions:

| Abstraction | Purpose | Extension point |
|---|---|---|
| `BenchmarkSuite` | Defines how tasks are loaded, run, and verified | `@register_suite` |
| `ContextStrategy` | SWE-bench context management approach | `@register_strategy` |
| `MemoryAdapter` | LENS memory system backend | `@register_adapter` |
| `BaseMetric` | Scoring metric (mechanical, LLM-judged, etc.) | `@register_metric` |

All four use decorator-based registries with lazy loading. See [docs/architecture.md](docs/architecture.md) for details.

## SWE-bench Strategies

| Strategy | Description |
|---|---|
| `naive` | Full message history, no management |
| `window` | Keep last N messages, drop oldest |
| `truncation` | Drop oldest until under token budget |
| `summary` | LLM-summarize every K steps |
| `masking` | Replace old tool outputs with placeholders |
| `rag` | BM25 retrieval from memory bank + recent window |
| `incremental_summary` | Rolling LLM compression after each step |
| `structured_summary` | JSON-schema-enforced periodic summarization |
| `hierarchical` | 3-tier system (hot/warm/cold) with masked warm + cold summary |
| `stack_heap` | Agent-controlled context via push/pop (stack) + alloc/write/free (heap) |

## LENS Adapters

| Adapter | Backend |
|---|---|
| `null` | No-op (baseline) |
| `sqlite` | SQLite full-text search |
| `sqlite-fts` | SQLite FTS5 |
| `sqlite-chunked` | Chunked document storage |
| `sqlite-embedding` | Local embedding similarity |
| `sqlite-embedding-openai` | OpenAI embedding similarity |
| `sqlite-hybrid` | FTS + embedding hybrid |
| `sqlite-hybrid-openai` | FTS + OpenAI embedding hybrid |
| `sqlite-chunked-hybrid` | Chunked + hybrid search |
| `mem0-raw` | [Mem0](https://mem0.ai/) raw storage |
| `mem0-extract` | Mem0 with extraction |
| `letta` | [Letta](https://letta.com/) agent memory |
| `letta-sleepy` | Letta with sleepy retention |
| `hindsight` | Hindsight consolidation |
| `graphiti` | [Graphiti](https://github.com/getzep/graphiti) knowledge graph |
| `cognee` | [Cognee](https://cognee.ai/) cognitive memory |
| `compaction` | LLM-driven compaction |

## Project Structure

```
synix-bench/
├── src/synix/
│   ├── core/              # Shared models, config, errors
│   ├── llm/               # LLM client abstraction (OpenAI-compatible)
│   ├── executor/          # Tool execution (container, local, modal)
│   ├── agent/             # Agent harness, budget enforcement, tool bridge
│   ├── suites/
│   │   ├── swebench/      # SWE-bench suite + 10 strategies
│   │   └── lens/          # LENS suite + 17 adapters
│   ├── scorer/            # 3-tier metric framework
│   ├── analysis/          # Bootstrap statistics, forest plots
│   ├── metering/          # HTTP proxy for token tracking
│   ├── viewer/            # Interactive trace viewer
│   ├── debug/             # Probe system for strategy debugging
│   ├── infra/             # Infrastructure providers (local, modal)
│   └── cli/               # Click CLI (7 commands)
├── configs/               # Layout configs, experiment configs
├── prompts/               # System prompts (stack-heap, manager)
├── datasets/              # LENS smoke test data
├── results/               # Append-only run output (gitignored)
└── tests/                 # 102 unit + conformance tests
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest -v

# Lint
uv run ruff check src/ tests/

# Smoke tests
synix-bench smoke --suite lens
synix-bench smoke --suite swebench
```

### Optional Dependencies

```bash
uv sync --extra scoring     # numpy, matplotlib (for analysis)
uv sync --extra swebench    # swebench library, docker SDK
uv sync --extra adapters    # mem0ai, letta-client
uv sync --extra all         # everything
```

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System architecture, core abstractions, data flow |
| [docs/configuration.md](docs/configuration.md) | Full configuration reference |
| [docs/strategies.md](docs/strategies.md) | SWE-bench strategy descriptions and parameters |
| [docs/adapters.md](docs/adapters.md) | LENS adapter interface and implementation guide |
| [docs/scoring.md](docs/scoring.md) | Metric framework and scoring pipeline |
| [docs/extending.md](docs/extending.md) | How to add suites, strategies, adapters, and metrics |

## License

MIT
