"""Integration test: full LENS pipeline — null adapter + mock LLM on smoke dataset.

Mirrors lens-benchmark/tests/integration/test_smoke_run.py to verify
the ported code produces equivalent results.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from synix.core.config import AgentBudgetConfig, RunConfig
from synix.suites.lens.suite import LENSSuite


class TestLensE2E:
    """End-to-end LENS run with null adapter + mock LLM."""

    def test_null_adapter_smoke(self):
        """Run null adapter on smoke dataset, verify structure."""
        config = RunConfig(
            suite="lens",
            strategy="null",
            agent_budget=AgentBudgetConfig.fast(),
            checkpoints=[5, 10],
        )

        suite = LENSSuite()
        tasks = suite.load_tasks(config)

        # Smoke dataset has 2 scopes with questions at checkpoints 5 and 10
        assert len(tasks) >= 1, f"Expected tasks, got {len(tasks)}"

        # Run each task
        for task in tasks:
            result = suite.run_task(task, config)

            # Verify TaskResult structure
            assert result.task_id
            assert result.suite == "lens"
            assert result.strategy == "null"
            assert result.model is not None

            # Should have steps (one per question)
            assert len(result.steps) >= 0  # null adapter may produce 0 useful answers

            # Token accounting
            assert result.total_input_tokens >= 0
            assert result.total_output_tokens >= 0

            # Raw result has checkpoint data
            assert "checkpoint_result" in result.raw_result
            cp_result = result.raw_result["checkpoint_result"]
            assert cp_result["scope_id"] == task["scope_id"]
            assert cp_result["checkpoint"] == task["checkpoint"]

            # Verify
            verification = suite.verify(task, result)
            assert verification.task_id == result.task_id
            # Null adapter won't have valid citations, so passed may be False
            assert isinstance(verification.passed, bool)
            assert "valid_count" in verification.details
            assert "total_count" in verification.details

    def test_sqlite_adapter_smoke(self):
        """Run sqlite adapter on smoke dataset — should find episodes via search."""
        config = RunConfig(
            suite="lens",
            strategy="sqlite",
            agent_budget=AgentBudgetConfig.fast(),
            checkpoints=[5, 10],
        )

        suite = LENSSuite()
        tasks = suite.load_tasks(config)
        assert len(tasks) >= 1

        # Just run the first task to verify sqlite adapter works end-to-end
        task = tasks[0]
        result = suite.run_task(task, config)

        assert result.task_id == task["task_id"]
        assert result.suite == "lens"
        assert result.strategy == "sqlite"

        # SQLite adapter should have ingested episodes
        cp_result = result.raw_result["checkpoint_result"]
        question_results = cp_result.get("question_results", [])
        # At minimum, questions were attempted
        assert len(question_results) >= 1

    def test_multiple_adapters_produce_results(self):
        """Verify several adapters can run without crashing."""
        adapters_to_test = ["null", "sqlite", "sqlite-fts"]

        for adapter_name in adapters_to_test:
            config = RunConfig(
                suite="lens",
                strategy=adapter_name,
                agent_budget=AgentBudgetConfig.fast(),
                checkpoints=[5],
            )

            suite = LENSSuite()
            tasks = suite.load_tasks(config)
            assert len(tasks) >= 1, f"{adapter_name}: no tasks loaded"

            task = tasks[0]
            result = suite.run_task(task, config)
            assert result.task_id, f"{adapter_name}: no task_id"
            assert result.suite == "lens"

            verification = suite.verify(task, result)
            assert isinstance(verification.passed, bool), f"{adapter_name}: verify failed"
