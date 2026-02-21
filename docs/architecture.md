# Architecture

synix-bench is a modular benchmark platform built around four core abstractions, each with a decorator-based plugin registry.

## System Overview

```
User
  │
  ▼
CLI (Click)
  │  synix-bench run --suite swebench --strategy naive ...
  │  synix-bench run --suite lens --strategy sqlite-hybrid ...
  ▼
RunEngine
  │  Dispatches to suite by name, collects TaskResults into SuiteResult
  │  Handles setup/teardown lifecycle, saves JSON results atomically
  ▼
BenchmarkSuite (ABC)
  │  load_tasks() → run_task() → verify()
  ├── SWEBenchSuite
  │     Loads SWE-bench instances, dispatches to ContextStrategy,
  │     runs agent in container, verifies patches against test suites
  │
  └── LENSSuite
        Streams episodes into MemoryAdapter, runs AgentHarness
        at checkpoints to answer questions, verifies citations
```

## Core Abstractions

### BenchmarkSuite

```python
class BenchmarkSuite(ABC):
    name: str

    def load_tasks(self, config: RunConfig) -> list[dict]: ...
    def run_task(self, task: dict, config: RunConfig) -> TaskResult: ...
    def verify(self, task: dict, result: TaskResult) -> VerificationResult: ...
    def setup(self, config: RunConfig) -> None: ...    # optional
    def teardown(self) -> None: ...                    # optional
    def list_strategies(self) -> list[str]: ...        # optional
```

Register with `@register_suite("name")`. The RunEngine looks up suites by name, handles the full lifecycle, and saves results.

### ContextStrategy (SWE-bench)

```python
class ContextStrategy(Protocol):
    name: str

    def run(self, client: OpenAI, model: str, task: str,
            executor: ToolExecutor, max_steps: int = 30, **kwargs) -> dict: ...
```

Each strategy implements a complete agent loop with its own context management approach. The `run()` method returns a dict with at minimum: `steps`, `patch`, `input_tokens`, `output_tokens`.

Register with `@register_strategy("name")`.

### MemoryAdapter (LENS)

```python
class MemoryAdapter(ABC):
    # Data loading (called by runner)
    def reset(self, scope_id: str) -> None: ...
    def ingest(self, episode_id, scope_id, timestamp, text, meta) -> None: ...
    def prepare(self, scope_id: str, checkpoint: int) -> None: ...

    # Tools (exposed to agent via tool bridge)
    def search(self, query, filters, limit) -> list[SearchResult]: ...
    def retrieve(self, ref_id: str) -> Document | None: ...
    def get_capabilities(self) -> CapabilityManifest: ...

    # Optional
    def get_synthetic_refs(self) -> list[tuple[str, str]]: ...
    def get_cache_state(self) -> dict | None: ...
    def restore_cache_state(self, state: dict) -> bool: ...
    def call_extended_tool(self, tool_name, arguments) -> object: ...
```

Register with `@register_adapter("name")`.

### BaseMetric (Scorer)

```python
class BaseMetric(ABC):
    @property
    def name(self) -> str: ...        # e.g. "evidence_validity"
    @property
    def tier(self) -> int: ...        # 1=mechanical, 2=LLM-judged, 3=future
    @property
    def description(self) -> str: ...

    def compute(self, result: RunResult) -> MetricResult: ...
```

Register with `@register_metric("name")`.

## Data Flow

### SWE-bench Flow

```
1. SWEBenchSuite.load_tasks()
   └── Loads SWE-bench instances (from dataset or sample)

2. SWEBenchSuite.run_task()
   ├── Build/start container (image_builder)
   ├── Create OpenAI client + ToolExecutor
   ├── Look up ContextStrategy by name
   └── strategy.run(client, model, task, executor)
       └── Agent loop: LLM call → parse tools → execute → manage context → repeat
       └── Returns: {steps, patch, input_tokens, output_tokens, ...}

3. SWEBenchSuite.verify()
   └── Run eval_script inside container, check fail_to_pass tests
```

### LENS Flow

```
1. LENSSuite.load_tasks()
   └── Load dataset → group episodes by scope → pair with checkpoint questions

2. LENSSuite.run_task()
   ├── Look up MemoryAdapter by name
   ├── adapter.reset(scope_id)
   ├── For each episode: adapter.ingest(...)
   ├── adapter.prepare(scope_id, checkpoint)
   ├── Register synthetic refs in EpisodeVault
   └── For each question:
       └── AgentHarness.answer(question, adapter)
           ├── Build tool definitions from adapter capabilities
           ├── LLM tool-calling loop (search, retrieve, extended tools)
           ├── Budget enforcement (turns, tool calls, tokens, latency)
           └── Return AgentAnswer with refs_cited

3. LENSSuite.verify()
   └── Check citations: answer must have text + at least one valid ref
```

## Unified Data Models

All suites produce results in the same schema:

```python
@dataclass
class StepTrace:
    step: int
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict]
    wall_time_ms: float
    extra: dict              # Suite-specific data

@dataclass
class TaskResult:
    task_id: str
    suite: str               # "swebench" or "lens"
    strategy: str            # Strategy/adapter name
    model: str
    steps: list[StepTrace]
    total_input_tokens: int
    total_output_tokens: int
    wall_time_s: float
    success: bool
    raw_result: dict         # Suite-specific deep data

@dataclass
class SuiteResult:
    suite: str
    strategy: str
    model: str
    tasks: list[TaskResult]
    config: dict
    # Properties: success_rate, total_tokens
```

Cross-suite analysis uses the flat fields. Suite-specific analysis (patch text, heap stats, ScoreCards) uses `raw_result`.

## Registries

All registries use the same pattern:

1. A module-level dict (`_REGISTRY`)
2. A `@register_X("name")` decorator that adds to the dict
3. A `get_X(name)` function with lazy loading
4. A `list_X()` function that returns all registered names

Lazy loading means strategy/adapter modules are imported on-demand when first looked up. This keeps startup fast and avoids import errors for optional dependencies (e.g., `mem0ai`, `letta-client`).

## Budget Enforcement

The agent harness enforces resource budgets during LENS runs:

| Budget | Default | Behavior |
|---|---|---|
| `max_turns` | 10 | Hard stop: raises `BudgetViolation` |
| `max_tool_calls` | 20 | Hard stop: raises `BudgetViolation` |
| `max_payload_bytes` | 64KB | Soft: logged, not raised |
| `max_latency_per_call_ms` | 5000 | Soft: logged, not raised |
| `max_agent_tokens` | 32768 | Soft: logged, not raised |

Budget presets: `fast`, `standard`, `extended`, `constrained-4k`, `constrained-2k`.

## Metering

The metering proxy intercepts OpenAI-compatible API calls to track adapter-internal token usage. This captures tokens spent by adapters that make their own LLM calls (e.g., compaction, mem0-extract).

```
Agent → Metering Proxy → Upstream LLM API
              │
              └── Records: model, prompt_tokens, completion_tokens, latency_ms
```

## Result Storage

Results are append-only JSON files written atomically to `results/`:

```
results/
  swebench_naive_20260220-143000.json
  lens_sqlite-hybrid_20260220-150000.json
```

Each file contains a full `SuiteResult` with all task results, traces, and the config used.
