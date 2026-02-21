"""Test StepTrace, TaskResult, SuiteResult serialization roundtrip."""
from __future__ import annotations

from synix.core.models import StepTrace, TaskResult, SuiteResult, VerificationResult


class TestStepTrace:
    def test_to_dict(self):
        step = StepTrace(
            step=0,
            input_tokens=100,
            output_tokens=50,
            tool_calls=[{"name": "bash", "args": {"cmd": "ls"}}],
            wall_time_ms=200.5,
            extra={"key": "val"},
        )
        d = step.to_dict()
        assert d["step"] == 0
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert len(d["tool_calls"]) == 1
        assert d["wall_time_ms"] == 200.5
        assert d["extra"]["key"] == "val"

    def test_from_dict(self):
        d = {
            "step": 3,
            "input_tokens": 200,
            "output_tokens": 100,
            "tool_calls": [],
            "wall_time_ms": 500.0,
        }
        step = StepTrace.from_dict(d)
        assert step.step == 3
        assert step.input_tokens == 200


class TestTaskResult:
    def test_roundtrip(self):
        tr = TaskResult(
            task_id="task-1",
            suite="lens",
            strategy="null",
            model="gpt-4o",
            steps=[
                StepTrace(step=0, input_tokens=10, output_tokens=5, tool_calls=[], wall_time_ms=100.0),
            ],
            total_input_tokens=10,
            total_output_tokens=5,
            wall_time_s=0.1,
            success=True,
            raw_result={"key": "value"},
        )
        d = tr.to_dict()
        restored = TaskResult.from_dict(d)
        assert restored.task_id == "task-1"
        assert restored.suite == "lens"
        assert restored.strategy == "null"
        assert restored.success is True
        assert len(restored.steps) == 1
        assert restored.raw_result["key"] == "value"


class TestVerificationResult:
    def test_roundtrip(self):
        vr = VerificationResult(
            task_id="task-1",
            passed=True,
            details={"score": 0.95},
        )
        d = vr.to_dict()
        restored = VerificationResult.from_dict(d)
        assert restored.task_id == "task-1"
        assert restored.passed is True
        assert restored.details["score"] == 0.95
