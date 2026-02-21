"""Truncation strategy: drop oldest messages until under token budget."""

from __future__ import annotations

import time

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.strategies.window import _run_agent_step
from synix.suites.swebench.tools import NAIVE_SYSTEM


@register_strategy("truncation")
class TruncationStrategy:
    """Token-budget truncation: estimate tokens, drop oldest until under budget."""

    name = "truncation"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        token_budget: int = 8000,
        **kwargs,
    ) -> dict:
        messages = [
            {"role": "system", "content": NAIVE_SYSTEM},
            {"role": "user", "content": task},
        ]

        trace = []
        totals = {
            "total_in": 0, "total_out": 0, "total_cached": 0, "total_managed": 0,
            "instruction_tokens": None, "stalls": 0,
        }
        t0 = time.monotonic()

        def _estimate_tokens(msg: dict) -> int:
            """Rough token estimate: len(content) // 4."""
            content = msg.get("content", "") or ""
            for tc in msg.get("tool_calls", []):
                content += tc.get("function", {}).get("arguments", "")
            return max(len(content) // 4, 1)

        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(messages)}")

            # Truncate: drop oldest messages after sys+user until under budget
            total_est = sum(_estimate_tokens(m) for m in messages)
            dropped = 0
            while total_est > token_budget and len(messages) > 2:
                removed = messages.pop(2)
                total_est -= _estimate_tokens(removed)
                dropped += 1
            if dropped:
                print(f"  (truncation: dropped {dropped} messages, est tokens now {total_est})")

            step_entry, should_stop = _run_agent_step(
                client, model, messages, executor, step_num, t0, totals, task,
            )
            if step_entry is None:
                break
            trace.append(step_entry)
            if should_stop:
                break

        elapsed = time.monotonic() - t0
        return {
            "trace": trace,
            "total_in": totals["total_in"],
            "total_out": totals["total_out"],
            "total_cached": totals["total_cached"],
            "total_managed": totals["total_managed"],
            "instruction_tokens": totals["instruction_tokens"] or 0,
            "elapsed_s": round(elapsed, 1),
        }
