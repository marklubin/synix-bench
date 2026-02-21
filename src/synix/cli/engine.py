"""RunEngine — dispatches to suites by name, collects results."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from synix.core.config import RunConfig
from synix.core.errors import SynixError, atomic_write
from synix.core.models import SuiteResult, TaskResult
from synix.suites.base import get_suite

log = logging.getLogger(__name__)


class RunEngine:
    """Top-level engine that dispatches to the appropriate BenchmarkSuite."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def run(self) -> SuiteResult:
        """Execute the full benchmark run."""
        suite_cls = get_suite(self.config.suite)
        suite = suite_cls()

        log.info("Starting run: suite=%s strategy=%s", self.config.suite, self.config.strategy)

        # Setup (e.g., prebuild images)
        suite.setup(self.config)

        try:
            # Load tasks
            tasks = suite.load_tasks(self.config)
            log.info("Loaded %d tasks", len(tasks))

            # Run each task
            results: list[TaskResult] = []
            for i, task in enumerate(tasks):
                task_id = task.get("instance_id", task.get("task_id", f"task-{i}"))
                log.info("Running task %d/%d: %s", i + 1, len(tasks), task_id)

                t0 = time.monotonic()
                try:
                    result = suite.run_task(task, self.config)
                    results.append(result)

                    # Verify
                    verification = suite.verify(task, result)
                    result.success = verification.passed
                    result.raw_result["verification"] = verification.to_dict()

                    elapsed = time.monotonic() - t0
                    log.info(
                        "Task %s: success=%s tokens=%d time=%.1fs",
                        task_id, result.success,
                        result.total_input_tokens + result.total_output_tokens,
                        elapsed,
                    )
                except SynixError as e:
                    log.error("Task %s failed: %s", task_id, e)
                    results.append(TaskResult(
                        task_id=task_id,
                        suite=self.config.suite,
                        strategy=self.config.strategy,
                        model=self.config.llm.model,
                        success=False,
                        raw_result={"error": str(e)},
                    ))
                except Exception as e:
                    log.error("Task %s unexpected error: %s", task_id, e, exc_info=True)
                    results.append(TaskResult(
                        task_id=task_id,
                        suite=self.config.suite,
                        strategy=self.config.strategy,
                        model=self.config.llm.model,
                        success=False,
                        raw_result={"error": str(e)},
                    ))

            suite_result = SuiteResult(
                suite=self.config.suite,
                strategy=self.config.strategy,
                model=self.config.llm.model,
                tasks=results,
                config=self.config.to_dict(),
            )

            # Save results
            self._save(suite_result)

            return suite_result
        finally:
            suite.teardown()

    def _save(self, result: SuiteResult) -> Path:
        """Save results to output directory."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{result.suite}_{result.strategy}_{timestamp}.json"
        out_path = out_dir / filename

        with atomic_write(out_path) as tmp:
            tmp.write_text(json.dumps(result.to_dict(), indent=2))

        log.info("Results saved to %s", out_path)
        return out_path
