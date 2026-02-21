# Extending synix-bench

synix-bench uses a plugin architecture with decorator-based registries. You can add new benchmark suites, context strategies, memory adapters, and scoring metrics without modifying core code.

## Adding a New Benchmark Suite

### 1. Create the suite module

```
src/synix/suites/mysuite/
  __init__.py
  suite.py
```

### 2. Implement the BenchmarkSuite ABC

```python
# src/synix/suites/mysuite/suite.py

from synix.core.config import RunConfig
from synix.core.models import TaskResult, VerificationResult
from synix.suites.base import BenchmarkSuite, register_suite


@register_suite("mysuite")
class MySuite(BenchmarkSuite):
    name = "mysuite"

    def load_tasks(self, config: RunConfig) -> list[dict]:
        """Load your benchmark tasks.

        Returns a list of dicts, each representing one task.
        The dict schema is suite-specific.
        """
        return [
            {"task_id": "task-1", "prompt": "..."},
            {"task_id": "task-2", "prompt": "..."},
        ]

    def run_task(self, task: dict, config: RunConfig) -> TaskResult:
        """Run a single task. This is where your agent loop goes."""
        # ... your logic ...
        return TaskResult(
            task_id=task["task_id"],
            suite="mysuite",
            strategy=config.strategy,
            model=config.llm.model,
            success=True,
            raw_result={"answer": "..."},
        )

    def verify(self, task: dict, result: TaskResult) -> VerificationResult:
        """Verify the result against ground truth."""
        passed = result.raw_result.get("answer") == task.get("expected")
        return VerificationResult(
            task_id=result.task_id,
            passed=passed,
            details={"expected": task.get("expected")},
        )
```

### 3. Register for lazy loading

Add your module to the lazy loader in `src/synix/suites/base.py`:

```python
def _lazy_load_suites() -> None:
    for module_name in [
        "synix.suites.swebench.suite",
        "synix.suites.lens.suite",
        "synix.suites.mysuite.suite",  # Add this
    ]:
        ...
```

### 4. Use it

```bash
synix-bench run --suite mysuite --strategy default
```

## Adding a New SWE-bench Strategy

### 1. Create the strategy file

```python
# src/synix/suites/swebench/strategies/my_strategy.py

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.tools import NAIVE_TOOLS, dispatch_tool_call


@register_strategy("my_strategy")
class MyStrategy:
    name = "my_strategy"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        **kwargs,
    ) -> dict:
        messages = [{"role": "system", "content": task}]
        total_in = 0
        total_out = 0
        steps = []

        for step in range(max_steps):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=NAIVE_TOOLS,
            )
            choice = response.choices[0]
            total_in += response.usage.prompt_tokens
            total_out += response.usage.completion_tokens

            # Check for tool calls
            if not choice.message.tool_calls:
                break

            # Execute tools
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                result = dispatch_tool_call(tc, executor)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # --- Your context management logic here ---
            messages = self._manage_context(messages)

            steps.append({
                "step": step,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            })

        return {
            "steps": steps,
            "patch": executor.get_patch(),
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    def _manage_context(self, messages):
        """Your context management logic."""
        return messages
```

### 2. Register for lazy loading

Add to `_lazy_load_strategies()` in `src/synix/suites/swebench/strategies/base.py`:

```python
"synix.suites.swebench.strategies.my_strategy",
```

### 3. Use it

```bash
synix-bench run --suite swebench --strategy my_strategy
```

## Adding a New LENS Adapter

### 1. Create the adapter file

```python
# src/synix/suites/lens/adapters/my_adapter.py

from synix.suites.lens.adapters.base import (
    CapabilityManifest, Document, MemoryAdapter, SearchResult,
)
from synix.suites.lens.adapters.registry import register_adapter


@register_adapter("my-adapter")
class MyAdapter(MemoryAdapter):
    def __init__(self):
        self._store: dict[str, dict] = {}

    def reset(self, scope_id: str) -> None:
        self._store.clear()

    def ingest(self, episode_id, scope_id, timestamp, text, meta=None) -> None:
        self._store[episode_id] = {
            "text": text,
            "timestamp": timestamp,
            "meta": meta or {},
        }

    def search(self, query, filters=None, limit=None) -> list[SearchResult]:
        limit = limit or 10
        results = []
        query_lower = query.lower()
        for ref_id, doc in self._store.items():
            if query_lower in doc["text"].lower():
                results.append(SearchResult(
                    ref_id=ref_id,
                    text=doc["text"][:500],
                    score=1.0,
                ))
        return results[:limit]

    def retrieve(self, ref_id: str) -> Document | None:
        doc = self._store.get(ref_id)
        if doc is None:
            return None
        return Document(ref_id=ref_id, text=doc["text"])

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["keyword"],
            max_results_per_search=10,
        )
```

### 2. Register for lazy loading

Add to the imports in `src/synix/suites/lens/adapters/registry.py`:

```python
_LAZY_MODULES = [
    ...
    "synix.suites.lens.adapters.my_adapter",
]
```

### 3. Verify with conformance tests

```bash
uv run pytest tests/conformance/test_adapter_conformance.py -v
```

### 4. Use it

```bash
synix-bench run --suite lens --strategy my-adapter
```

## Adding a New Metric

### 1. Create the metric

```python
# src/synix/scorer/my_metric.py

from synix.scorer.base import BaseMetric
from synix.scorer.registry import register_metric
from synix.suites.lens.models import MetricResult, RunResult


@register_metric("my_metric")
class MyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "my_metric"

    @property
    def tier(self) -> int:
        return 1  # Mechanical (no LLM needed)

    @property
    def description(self) -> str:
        return "Measures something useful about the run"

    def compute(self, result: RunResult) -> MetricResult:
        # Access result data
        total_questions = 0
        good_answers = 0

        for cp in result.checkpoints:
            for qr in cp.question_results:
                total_questions += 1
                if qr.answer and qr.answer.answer_text:
                    good_answers += 1

        value = good_answers / max(total_questions, 1)

        return MetricResult(
            name=self.name,
            value=value,
            tier=self.tier,
            details={
                "good_answers": good_answers,
                "total_questions": total_questions,
            },
            sample_size=total_questions,
        )
```

### 2. Register for lazy loading

Add to the metric registry's lazy loader in `src/synix/scorer/registry.py`.

### 3. Use it

```bash
synix-bench score --results results/*.json
# Your metric will appear in the scorecard automatically
```

## Plugin Architecture Notes

- **Lazy loading**: All registries use lazy imports. Modules are only loaded when first accessed. This keeps startup fast and avoids import errors for optional dependencies.
- **Optional dependencies**: If your extension needs packages not in the core requirements, users install them via extras: `uv sync --extra adapters`.
- **Entry points**: The `pyproject.toml` defines entry point groups for `synix.adapters` and `synix.metrics`. External packages can register implementations through these entry points (not yet automatically discovered, but the infrastructure is in place).
- **No core modifications**: A well-designed extension only adds new files and a one-line entry in the lazy loader. No existing code needs to change.
