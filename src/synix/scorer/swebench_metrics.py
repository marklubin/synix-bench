"""SWE-bench specific metrics using the scorer framework.

These metrics operate on synix.core.models.TaskResult (cross-suite model)
rather than LENS RunResult. They use the same BaseMetric ABC and registry
but extract data from TaskResult.raw_result.
"""
from __future__ import annotations

from synix.core.models import TaskResult
from synix.suites.lens.models import MetricResult
from synix.scorer.base import BaseMetric
from synix.scorer.registry import register_metric


def _compute_from_task_results(
    task_results: list[TaskResult],
    extractor,
) -> tuple[float, dict]:
    """Helper: apply extractor to each task result and aggregate.

    extractor(task_result) -> float | None (None = skip)
    Returns (mean_value, details_dict).
    """
    values: list[float] = []
    for tr in task_results:
        v = extractor(tr)
        if v is not None:
            values.append(v)

    if not values:
        return 0.0, {"n_tasks": 0}

    mean_val = sum(values) / len(values)
    return mean_val, {"n_tasks": len(values)}


@register_metric("verifier_pass_rate")
class VerifierPassRate(BaseMetric):
    """Fraction of tasks where the verifier confirmed the patch is correct.

    Reads raw_result.get("verifier_pass") from each TaskResult.
    This is the gold-standard success metric for SWE-bench: the patch
    must pass the fail_to_pass test cases.

    Note: This metric does NOT use RunResult — it must be called via
    compute_from_tasks() rather than compute().
    """

    @property
    def name(self) -> str:
        return "verifier_pass_rate"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Fraction of tasks where verifier_pass == True"

    def compute(self, result) -> MetricResult:
        """Stub for RunResult-based interface. Use compute_from_tasks instead."""
        return MetricResult(
            name=self.name, tier=self.tier, value=0.0,
            details={"error": "Use compute_from_tasks() for SWE-bench metrics"},
        )

    def compute_from_tasks(self, task_results: list[TaskResult]) -> MetricResult:
        """Compute verifier pass rate from a list of TaskResults."""
        def _extract(tr: TaskResult) -> float | None:
            return 1.0 if tr.raw_result.get("verifier_pass") is True else 0.0

        value, details = _compute_from_task_results(task_results, _extract)
        n_passed = sum(
            1 for tr in task_results
            if tr.raw_result.get("verifier_pass") is True
        )
        details["n_passed"] = n_passed
        return MetricResult(name=self.name, tier=self.tier, value=value, details=details)


@register_metric("patch_rate")
class PatchRate(BaseMetric):
    """Fraction of tasks that produced a non-empty patch.

    This is a softer success metric than verifier_pass_rate: it counts
    whether the agent produced any diff at all, regardless of correctness.
    Reads raw_result.get("has_patch") or checks for non-empty "patch" field.
    """

    @property
    def name(self) -> str:
        return "patch_rate"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Fraction of tasks that produced a non-empty patch"

    def compute(self, result) -> MetricResult:
        """Stub for RunResult-based interface. Use compute_from_tasks instead."""
        return MetricResult(
            name=self.name, tier=self.tier, value=0.0,
            details={"error": "Use compute_from_tasks() for SWE-bench metrics"},
        )

    def compute_from_tasks(self, task_results: list[TaskResult]) -> MetricResult:
        """Compute patch production rate from a list of TaskResults."""
        def _extract(tr: TaskResult) -> float | None:
            has_patch = tr.raw_result.get("has_patch", False)
            if not has_patch:
                # Fallback: check if "patch" field is non-empty
                patch_text = tr.raw_result.get("patch", "")
                has_patch = bool(patch_text and patch_text.strip())
            return 1.0 if has_patch else 0.0

        value, details = _compute_from_task_results(task_results, _extract)
        n_patched = sum(
            1 for tr in task_results
            if tr.raw_result.get("has_patch", False)
            or bool(tr.raw_result.get("patch", "").strip())
        )
        details["n_patched"] = n_patched
        return MetricResult(name=self.name, tier=self.tier, value=value, details=details)


@register_metric("token_efficiency")
class TokenEfficiency(BaseMetric):
    """Success rate per million tokens consumed.

    Computed as: success_rate / (total_tokens / 1_000_000).
    Higher is better — an agent that solves more tasks with fewer tokens
    is more efficient. Uses verifier_pass as the success criterion.

    Returns 0.0 if no tokens were consumed (avoids division by zero).
    """

    @property
    def name(self) -> str:
        return "token_efficiency"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Success rate per million tokens (verifier_pass / M_tokens)"

    def compute(self, result) -> MetricResult:
        """Stub for RunResult-based interface. Use compute_from_tasks instead."""
        return MetricResult(
            name=self.name, tier=self.tier, value=0.0,
            details={"error": "Use compute_from_tasks() for SWE-bench metrics"},
        )

    def compute_from_tasks(self, task_results: list[TaskResult]) -> MetricResult:
        """Compute token efficiency from a list of TaskResults."""
        if not task_results:
            return MetricResult(
                name=self.name, tier=self.tier, value=0.0,
                details={"n_tasks": 0},
            )

        n_passed = sum(
            1 for tr in task_results
            if tr.raw_result.get("verifier_pass") is True
        )
        success_rate = n_passed / len(task_results)

        total_tokens = sum(
            tr.total_input_tokens + tr.total_output_tokens
            for tr in task_results
        )
        m_tokens = total_tokens / 1_000_000

        if m_tokens == 0:
            value = 0.0
        else:
            value = success_rate / m_tokens

        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "n_tasks": len(task_results),
                "n_passed": n_passed,
                "success_rate": success_rate,
                "total_tokens": total_tokens,
                "m_tokens": round(m_tokens, 3),
            },
        )
