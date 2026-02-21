from __future__ import annotations

import logging
from typing import Callable

from synix.suites.lens.models import MetricResult, RunResult, ScoreCard
from synix.scorer.aggregate import build_scorecard
from synix.scorer.registry import list_metrics

logger = logging.getLogger(__name__)


class ScorerEngine:
    """Runs all registered metrics against a run result and produces a ScoreCard.

    Supports optional judge_fn for metrics that require LLM-based evaluation
    (e.g., pairwise answer_quality). Metrics that implement configure() will
    receive the judge_fn and any extra kwargs automatically.
    """

    def __init__(
        self,
        tier_filter: int | None = None,
        judge_fn: Callable[[str], str] | None = None,
        gate_thresholds: dict[str, float] | None = None,
        baseline_generator=None,
        max_judge_workers: int = 1,
    ) -> None:
        self.tier_filter = tier_filter
        self.judge_fn = judge_fn
        self.gate_thresholds = gate_thresholds
        self.baseline_generator = baseline_generator
        self.max_judge_workers = max_judge_workers

    def score(self, result: RunResult) -> ScoreCard:
        """Score a run result with all applicable metrics."""
        all_metrics = list_metrics()
        results: list[MetricResult] = []

        # Build configure kwargs
        configure_kwargs: dict = {}
        if self.judge_fn is not None:
            configure_kwargs["judge_fn"] = self.judge_fn
        if self.baseline_generator is not None:
            configure_kwargs["baseline_generator"] = self.baseline_generator
        if self.max_judge_workers > 1:
            configure_kwargs["max_judge_workers"] = self.max_judge_workers

        for name, metric_cls in sorted(all_metrics.items()):
            metric = metric_cls()

            # Inject configuration for metrics that support it
            if configure_kwargs and hasattr(metric, "configure"):
                metric.configure(**configure_kwargs)

            if self.tier_filter is not None and metric.tier != self.tier_filter:
                continue

            logger.info("Computing %s (tier %d)", name, metric.tier)
            metric_result = metric.compute(result)
            results.append(metric_result)
            logger.info("  %s = %.4f", name, metric_result.value)

        scorecard = build_scorecard(
            run_id=result.run_id,
            adapter=result.adapter,
            dataset_version=result.dataset_version,
            budget_preset=result.budget_preset,
            metrics=results,
            gate_thresholds=self.gate_thresholds,
        )

        logger.info("Composite score: %.4f", scorecard.composite_score)
        return scorecard
