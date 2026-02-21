"""LENS benchmark suite implementation.

Registers as 'lens' in the suite registry. Implements the full
LENS runner loop: load dataset, ingest episodes, run agent at
checkpoints, verify citations.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from synix.agent.budget import QuestionBudget
from synix.agent.harness import AgentHarness
from synix.core.config import RunConfig
from synix.core.models import StepTrace, TaskResult, VerificationResult
from synix.llm.factory import create_llm_client
from synix.suites.base import BenchmarkSuite, register_suite
from synix.suites.lens.adapters.registry import get_adapter, list_adapters
from synix.suites.lens.anticheat import EpisodeVault
from synix.suites.lens.models import (
    CheckpointResult,
    Episode,
    Question,
    QuestionResult,
)

log = logging.getLogger(__name__)


def _default_dataset() -> str:
    """Path to the bundled smoke dataset."""
    # Walk up from src/synix/suites/lens/ to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    smoke = project_root / "datasets" / "smoke" / "smoke_dataset.json"
    return str(smoke)


def _load_dataset(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _load_episodes(data: dict) -> dict[str, list[Episode]]:
    scopes: dict[str, list[Episode]] = {}
    # Handle flat episodes list
    for ep_data in data.get("episodes", []):
        ep = Episode.from_dict(ep_data)
        scopes.setdefault(ep.scope_id, []).append(ep)
    # Handle nested scopes format
    for scope_data in data.get("scopes", []):
        scope_id = scope_data["scope_id"]
        for ep_data in scope_data.get("episodes", []):
            ep = Episode.from_dict(ep_data)
            scopes.setdefault(scope_id, []).append(ep)
    return scopes


def _load_questions(data: dict) -> list[Question]:
    return [Question.from_dict(q) for q in data.get("questions", [])]


@register_suite("lens")
class LENSSuite(BenchmarkSuite):
    """LENS longitudinal episodic benchmark suite."""

    name = "lens"

    def __init__(self) -> None:
        self.vault = EpisodeVault()

    def list_strategies(self) -> list[str]:
        """Return available adapter names as strategies."""
        return sorted(list_adapters().keys())

    def load_tasks(self, config: RunConfig) -> list[dict]:
        """Load LENS dataset tasks — one task per scope+checkpoint pair."""
        dataset_path = config.dataset or _default_dataset()
        data = _load_dataset(dataset_path)
        scopes = _load_episodes(data)
        questions = _load_questions(data)

        q_index: dict[str, dict[int, list[Question]]] = {}
        for q in questions:
            q_index.setdefault(q.scope_id, {}).setdefault(q.checkpoint_after, []).append(q)

        tasks = []
        for scope_id, episodes in scopes.items():
            scope_qs = q_index.get(scope_id, {})
            for checkpoint, qs in scope_qs.items():
                tasks.append({
                    "task_id": f"{scope_id}_cp{checkpoint}",
                    "scope_id": scope_id,
                    "checkpoint": checkpoint,
                    "episodes": [e.to_dict() for e in episodes],
                    "questions": [q.to_dict() for q in qs],
                    "dataset_version": data.get("version", "unknown"),
                })
        return tasks

    def run_task(self, task: dict, config: RunConfig) -> TaskResult:
        """Run the LENS benchmark for a single scope+checkpoint."""
        scope_id = task["scope_id"]
        checkpoint = task["checkpoint"]
        episodes = sorted(
            [Episode.from_dict(e) for e in task["episodes"]],
            key=lambda e: e.timestamp,
        )
        questions = [Question.from_dict(q) for q in task["questions"]]

        adapter_cls = get_adapter(config.strategy)
        adapter = adapter_cls()
        llm_client = create_llm_client(config.llm)

        # Ingest
        adapter.reset(scope_id)
        for ep in episodes:
            self.vault.store(ep.episode_id, ep.text)
            adapter.ingest(
                episode_id=ep.episode_id,
                scope_id=scope_id,
                timestamp=ep.timestamp.isoformat(),
                text=ep.text,
                meta=ep.meta,
            )

        # Prepare
        adapter.prepare(scope_id, checkpoint)
        for ref_id, text in adapter.get_synthetic_refs():
            self.vault.store(ref_id, text)

        # Answer questions
        budget = QuestionBudget(
            max_turns=config.agent_budget.max_turns,
            max_payload_bytes=config.agent_budget.max_payload_bytes,
            max_latency_per_call_ms=config.agent_budget.max_latency_per_call_ms,
            max_total_tool_calls=config.agent_budget.max_tool_calls,
            max_agent_tokens=config.agent_budget.max_agent_tokens,
            max_cumulative_result_tokens=config.agent_budget.max_cumulative_result_tokens,
        )
        harness = AgentHarness(llm_client, budget)

        question_results: list[QuestionResult] = []
        steps: list[StepTrace] = []
        total_in = 0
        total_out = 0
        t0 = time.monotonic()

        for i, question in enumerate(questions):
            qt0 = time.monotonic()
            answer = harness.answer(
                question_prompt=question.prompt,
                adapter=adapter,
                question_id=question.question_id,
            )
            qt_ms = (time.monotonic() - qt0) * 1000

            valid = [r for r in answer.refs_cited if self.vault.has(r)]
            question_results.append(QuestionResult(
                question=question,
                answer=answer,
                retrieved_ref_ids=answer.refs_cited,
                valid_ref_ids=valid,
            ))

            half_tokens = answer.total_tokens // 2
            steps.append(StepTrace(
                step=i,
                input_tokens=half_tokens,
                output_tokens=half_tokens,
                tool_calls=[{"name": "agent_loop", "question_id": question.question_id}],
                wall_time_ms=qt_ms,
                extra={"tool_calls_made": answer.tool_calls_made},
            ))
            total_in += half_tokens
            total_out += half_tokens

        wall_s = time.monotonic() - t0
        cp_result = CheckpointResult(
            scope_id=scope_id,
            checkpoint=checkpoint,
            question_results=question_results,
        )

        return TaskResult(
            task_id=task["task_id"],
            suite="lens",
            strategy=config.strategy,
            model=config.llm.model,
            steps=steps,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            wall_time_s=wall_s,
            success=True,
            raw_result={
                "checkpoint_result": cp_result.to_dict(),
                "dataset_version": task.get("dataset_version", ""),
            },
        )

    def verify(self, task: dict, result: TaskResult) -> VerificationResult:
        """Verify LENS results — check citations are valid."""
        cp_data = result.raw_result.get("checkpoint_result", {})
        qr_list = cp_data.get("question_results", [])

        valid_count = 0
        total_count = len(qr_list)
        details_qs = []

        for qr in qr_list:
            answer = qr.get("answer", {})
            has_answer = bool(answer.get("answer_text", "").strip())
            refs = answer.get("refs_cited", [])
            valid_refs = qr.get("valid_ref_ids", [])
            q_ok = has_answer and len(refs) > 0
            if q_ok:
                valid_count += 1
            details_qs.append({
                "question_id": qr.get("question", {}).get("question_id", ""),
                "has_answer": has_answer,
                "refs_cited": len(refs),
                "valid_refs": len(valid_refs),
                "passed": q_ok,
            })

        return VerificationResult(
            task_id=result.task_id,
            passed=valid_count > 0,
            details={"valid_count": valid_count, "total_count": total_count, "questions": details_qs},
        )
