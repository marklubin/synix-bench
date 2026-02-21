"""SWE-bench benchmark suite.

Implements BenchmarkSuite ABC: load SWE-bench instances, dispatch to a
ContextStrategy, verify patches via eval scripts.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from openai import OpenAI

from synix.core.config import RunConfig
from synix.core.models import TaskResult, VerificationResult
from synix.suites.base import BenchmarkSuite, register_suite
from synix.suites.swebench.image_builder import (
    ContainerExecutor,
    build_instance_image,
    start_container,
    stop_container,
)
from synix.suites.swebench.strategies._common import parse_pytest_output
from synix.suites.swebench.strategies.base import get_strategy, list_strategies

log = logging.getLogger(__name__)


def build_task_description(instance: dict) -> str:
    """Build the task prompt from a SWE-bench instance."""
    return (
        f"Fix the following issue in the {instance['repo']} repository:\n\n"
        f"{instance['problem_statement']}\n\n"
        f"The full repository is at /testbed (already cloned, deps installed). "
        f"Organize your work into phases using push_frame/pop_frame:\n"
        f"1. Investigate: find the relevant source files and understand the bug. "
        f"Use heap_alloc to store key findings (file paths, code snippets, root cause) "
        f"so the next phase doesn't re-read files.\n"
        f"2. Fix: implement the code change. heap_read your investigation notes.\n"
        f"3. Verify: run relevant tests to confirm the fix (e.g., pytest path/to/test.py -x).\n\n"
        f"pop_frame each phase when done. The result string is ALL the next phase sees."
    )


def load_dataset_instances(
    instance_ids: list[str] | None = None,
    sample: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load SWE-bench Verified instances from HuggingFace."""
    from datasets import load_dataset

    log.info("Loading SWE-bench Verified from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    log.info("Loaded %d instances", len(ds))

    if instance_ids:
        instances = [row for row in ds if row["instance_id"] in instance_ids]
        missing = set(instance_ids) - {row["instance_id"] for row in instances}
        if missing:
            log.warning("Instance IDs not found in dataset: %s", missing)
        return instances

    if sample:
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(sample, len(ds)))
        return [ds[i] for i in indices]

    return list(ds)


@register_suite("swebench")
class SWEBenchSuite(BenchmarkSuite):
    """SWE-bench Verified benchmark suite."""

    name = "swebench"

    def load_tasks(self, config: RunConfig) -> list[dict]:
        """Load SWE-bench instances based on config."""
        if config.instance_ids:
            instance_ids = config.instance_ids
        elif config.instance_id:
            instance_ids = [config.instance_id]
        else:
            instance_ids = None
        return load_dataset_instances(
            instance_ids=instance_ids,
            sample=config.sample,
            seed=config.seed,
        )

    def run_task(self, task: dict, config: RunConfig) -> TaskResult:
        """Run a single SWE-bench instance through a strategy."""
        instance_id = task["instance_id"]
        strategy_name = config.strategy
        task_desc = build_task_description(task)

        log.info("Running %s on %s with strategy %s", self.name, instance_id, strategy_name)

        # Build image and start container
        image = build_instance_image(task)
        cid = start_container(image)
        executor = ContainerExecutor(cid)

        try:
            # Get strategy class and instantiate
            strategy_cls = get_strategy(strategy_name)
            strategy = strategy_cls()

            # Build client
            client = OpenAI(
                api_key=config.llm.api_key or "dummy",
                base_url=config.llm.api_base,
            )

            # Load layout config if specified (for stack_heap strategy)
            layout = None
            if config.swebench.layout_file:
                import json as _json
                from pathlib import Path as _Path

                layout = _json.loads(_Path(config.swebench.layout_file).read_text())

            # Run the strategy
            run_result = strategy.run(
                client=client,
                model=config.llm.model,
                task=task_desc,
                executor=executor,
                max_steps=config.swebench.max_steps,
                layout=layout,
                no_think_prefill=config.swebench.no_think_prefill,
            )

            # Extract patch
            patch = executor.get_patch()

            trace = run_result.get("trace", [])
            return TaskResult(
                task_id=instance_id,
                suite=self.name,
                strategy=strategy_name,
                model=config.llm.model,
                total_input_tokens=run_result.get("total_in", 0),
                total_output_tokens=run_result.get("total_out", 0),
                wall_time_s=run_result.get("elapsed_s", 0),
                success=bool(patch),
                raw_result={
                    "patch": patch,
                    "trace": trace,
                    "total_cached": run_result.get("total_cached", 0),
                    "total_managed": run_result.get("total_managed", 0),
                    "instruction_tokens": run_result.get("instruction_tokens", 0),
                },
            )
        finally:
            stop_container(cid)

    def verify(self, task: dict, result: TaskResult) -> VerificationResult:
        """Verify a task result by running the SWE-bench eval script."""
        instance_id = task["instance_id"]
        patch = result.raw_result.get("patch", "")

        if not patch:
            return VerificationResult(
                task_id=instance_id,
                passed=False,
                details={"error": "No patch produced"},
            )

        # Build image, start container, apply patch, run eval
        try:
            image = build_instance_image(task)
            cid = start_container(image)
            executor = ContainerExecutor(cid)

            try:
                from swebench.harness.test_spec.test_spec import make_test_spec
                spec = make_test_spec(task)
                eval_script = spec.eval_script
                executor._exec(
                    "cat > /tmp/eval.sh << 'EVALEOF'\n" + eval_script + "\nEVALEOF",
                    timeout=10,
                )
                executor._exec("chmod +x /tmp/eval.sh", timeout=10)
                test_output = executor._exec("bash /tmp/eval.sh 2>&1", timeout=600)
                parsed = parse_pytest_output(test_output)

                details = {**parsed, "test_output": test_output[:5000]}

                fail_to_pass = spec.FAIL_TO_PASS if hasattr(spec, "FAIL_TO_PASS") else []
                if fail_to_pass:
                    details["fail_to_pass"] = fail_to_pass

                return VerificationResult(
                    task_id=instance_id,
                    passed=parsed.get("all_passed", False),
                    details=details,
                )
            finally:
                stop_container(cid)
        except Exception as e:
            log.error("Verification failed for %s: %s", instance_id, e)
            return VerificationResult(
                task_id=instance_id,
                passed=False,
                details={"error": str(e)},
            )

    def setup(self, config: RunConfig) -> None:
        """Pre-build container images if requested."""
        if config.swebench.prebuild_only:
            tasks = self.load_tasks(config)
            log.info("Pre-building images for %d instances...", len(tasks))
            for task in tasks:
                try:
                    build_instance_image(task)
                except Exception as e:
                    log.error("Failed to build image for %s: %s", task["instance_id"], e)

    def list_strategies(self) -> list[str]:
        """Return available strategy names."""
        return list_strategies()
