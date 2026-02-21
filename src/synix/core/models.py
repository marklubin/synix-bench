"""Unified data models for synix-bench.

Contains both shared cross-suite models (StepTrace, TaskResult) and
LENS-specific models (Episode, Question, etc.) that live under suites/lens/
but are re-exported here for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Cross-suite unified models
# ---------------------------------------------------------------------------


@dataclass
class StepTrace:
    """A single step in an agent's execution trace."""

    step: int
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
    wall_time_ms: float = 0.0
    extra: dict = field(default_factory=dict)  # suite-specific (heap stats, registers, etc.)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "wall_time_ms": self.wall_time_ms,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepTrace:
        return cls(
            step=d["step"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            tool_calls=d.get("tool_calls", []),
            wall_time_ms=d.get("wall_time_ms", 0.0),
            extra=d.get("extra", {}),
        )


@dataclass
class TaskResult:
    """Result of running one task through one suite+strategy/adapter.

    The flat fields (task_id, suite, strategy, model, steps, tokens, timing,
    success) are used for cross-suite analysis. Suite-specific data lives in
    raw_result (HMB patch text, heap stats; LENS ScoreCard, episode refs).
    """

    task_id: str
    suite: str  # "swebench" or "lens"
    strategy: str  # strategy name (swebench) or adapter name (lens)
    model: str
    steps: list[StepTrace] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_time_s: float = 0.0
    success: bool = False
    raw_result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "suite": self.suite,
            "strategy": self.strategy,
            "model": self.model,
            "steps": [s.to_dict() for s in self.steps],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "wall_time_s": self.wall_time_s,
            "success": self.success,
            "raw_result": self.raw_result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskResult:
        return cls(
            task_id=d["task_id"],
            suite=d["suite"],
            strategy=d["strategy"],
            model=d["model"],
            steps=[StepTrace.from_dict(s) for s in d.get("steps", [])],
            total_input_tokens=d.get("total_input_tokens", 0),
            total_output_tokens=d.get("total_output_tokens", 0),
            wall_time_s=d.get("wall_time_s", 0.0),
            success=d.get("success", False),
            raw_result=d.get("raw_result", {}),
        )


@dataclass
class SuiteResult:
    """Aggregated results for a full suite run (multiple tasks)."""

    suite: str
    strategy: str
    model: str
    tasks: list[TaskResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.success) / len(self.tasks)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_input_tokens + t.total_output_tokens for t in self.tasks)

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "strategy": self.strategy,
            "model": self.model,
            "tasks": [t.to_dict() for t in self.tasks],
            "config": self.config,
            "success_rate": self.success_rate,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SuiteResult:
        return cls(
            suite=d["suite"],
            strategy=d["strategy"],
            model=d["model"],
            tasks=[TaskResult.from_dict(t) for t in d.get("tasks", [])],
            config=d.get("config", {}),
        )


@dataclass
class VerificationResult:
    """Result of verifying a task's output."""

    task_id: str
    passed: bool
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerificationResult:
        return cls(
            task_id=d["task_id"],
            passed=d["passed"],
            details=d.get("details", {}),
        )
