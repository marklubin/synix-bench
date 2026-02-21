# Scoring Framework

synix-bench includes a tiered metric framework for evaluating benchmark results. Metrics are organized into tiers by complexity and cost.

## Metric Tiers

### Tier 1: Mechanical Metrics

Pure computation, no LLM calls. Fast and deterministic.

| Metric | Description |
|---|---|
| `evidence_validity` | Fraction of cited references that exist in the EpisodeVault |
| `citation_density` | Average number of citations per answer |
| `temporal_coverage` | How well answers cover the temporal range of relevant episodes |
| `response_completeness` | Whether questions have non-empty answers with citations |
| `retrieval_precision` | Fraction of retrieved results actually used in answers |
| `budget_utilization` | How efficiently the agent uses its turn/tool budget |

### Tier 2: LLM-Judged Metrics

Require an LLM judge function. Higher cost but measure semantic quality.

| Metric | Description |
|---|---|
| `answer_quality` | LLM-judged quality of answers on a 1-5 scale |
| `contradiction_check` | Whether answers contradict known ground truth |

### Tier 3: Experimental Metrics

Future metrics under development.

| Metric | Description |
|---|---|
| `temporal_reasoning` | Quality of temporal reasoning in multi-episode questions |

### SWE-bench Metrics

Additional metrics for the SWE-bench suite:

| Metric | Description |
|---|---|
| `verifier_pass_rate` | Fraction of tasks passing verification (test suite) |
| `patch_rate` | Fraction of tasks producing a non-empty patch |
| `token_efficiency` | Verification pass rate normalized by token usage |

## Using the Scorer

### CLI

```bash
# Score all metrics
synix-bench score --results results/lens_sqlite-hybrid_*.json

# Score only tier 1 (mechanical, no LLM needed)
synix-bench score --results results/lens_sqlite-hybrid_*.json --tier 1
```

### Programmatic

```python
from synix.scorer.engine import ScorerEngine
from synix.suites.lens.models import RunResult

# Load a run result
result = RunResult.from_dict(...)

# Score with mechanical metrics only
engine = ScorerEngine(tier_filter=1)
scorecard = engine.score(result)

print(f"Composite score: {scorecard.composite_score:.4f}")
for metric in scorecard.metrics:
    print(f"  {metric.name}: {metric.value:.4f}")

# Score with LLM judge
def judge(prompt: str) -> str:
    # Your LLM call here
    return response

engine = ScorerEngine(judge_fn=judge)
scorecard = engine.score(result)
```

## ScoreCard

The `ScorerEngine.score()` method returns a `ScoreCard`:

```python
@dataclass
class ScoreCard:
    run_id: str
    adapter: str
    dataset_version: str
    budget_preset: str
    metrics: list[MetricResult]
    composite_score: float         # Weighted average of all metrics
    composite_with_gate: float     # Score after applying gate thresholds
    gate_pass: bool                # Whether all gates passed
    timestamp: str
```

Each `MetricResult` contains:

```python
@dataclass
class MetricResult:
    name: str
    value: float          # 0.0 to 1.0
    tier: int
    details: dict         # Metric-specific breakdown
    sample_size: int
```

## Gate Thresholds

Gates are minimum score thresholds. If any gated metric falls below its threshold, `gate_pass` is `False` and `composite_with_gate` is penalized.

```python
engine = ScorerEngine(
    gate_thresholds={
        "evidence_validity": 0.3,
        "response_completeness": 0.5,
    }
)
```

## Adding Custom Metrics

See [extending.md](extending.md) for how to implement and register new metrics.

### Quick Example

```python
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
        return 1

    @property
    def description(self) -> str:
        return "My custom metric"

    def compute(self, result: RunResult) -> MetricResult:
        # Your computation here
        value = 0.85
        return MetricResult(
            name=self.name,
            value=value,
            tier=self.tier,
            details={"custom_field": "data"},
            sample_size=len(result.checkpoints),
        )
```
