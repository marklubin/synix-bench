# synix-bench

Unified benchmark platform for agent memory and context management strategies.
Combines hybrid-memory-bench (SWE-bench coding tasks) and lens-benchmark (longitudinal reasoning).

## Architecture

```
synix-bench run --suite swebench --strategy naive ...
synix-bench run --suite lens --adapter sqlite-hybrid ...
synix-bench sweep --config matrix.json
synix-bench score / synix-bench report / synix-bench compare
```

Two benchmark suites sharing core infrastructure:
- **swebench**: 10 context management strategies on SWE-bench coding tasks
- **lens**: 21+ memory adapters on longitudinal reasoning scenarios

## Directory Layout

```
src/synix/
  core/          # Shared models, config, errors
  llm/           # LLM client abstraction (OpenAI-compatible)
  executor/      # Tool execution (container, local, modal stub)
  agent/         # Agent harness, budget enforcement, tool bridge
  suites/
    swebench/    # SWE-bench suite + 10 strategies
    lens/        # LENS suite + adapters
  scorer/        # Metric framework (3 tiers)
  viewer/        # Interactive trace viewer
  debug/         # Probe system
  analysis/      # Bootstrap stats, plots
  metering/      # HTTP proxy for token tracking
  infra/         # Infrastructure providers (local, modal stub)
  cli/           # Click CLI
  datagen/       # LENS dataset generation
configs/         # Layout configs, experiment configs
prompts/         # System prompts (stack-heap, manager)
datasets/        # LENS scopes, smoke test data
results/         # Append-only run output
tests/           # Unit, integration, conformance
```

## Conventions

- **uv-native** (no pip, no setup.py)
- **Config-driven.** Every knob is a key in config. No knob requires a code change.
- **Results are append-only.** Every run writes a self-contained JSON to `results/`.
- **Prompts are text files.** System prompts live in `prompts/` as plain text.
- **Never silently swallow errors.** Log failures with detail.
- **Plugin architecture.** Adapters and metrics use registry decorators.

## Coding Style

- Type hints on function signatures
- Flat config dicts, no unnecessary classes
- One ABC per extension point (BenchmarkSuite, ContextStrategy, MemoryAdapter, BaseMetric)
- Everything else is functions
