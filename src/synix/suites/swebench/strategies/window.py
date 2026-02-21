"""Window strategy: sliding window context management."""

from __future__ import annotations

import json
import time

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.strategies._common import (
    DONE_PHRASES,
    extract_usage,
    make_step_entry,
    process_tool_calls,
)
from synix.suites.swebench.tools import NAIVE_SYSTEM, NAIVE_TOOLS


def _run_agent_step(
    client: OpenAI,
    model: str,
    messages: list[dict],
    executor: ToolExecutor,
    step_num: int,
    t0: float,
    totals: dict,
    task: str = "(task)",
) -> tuple[dict | None, bool]:
    """Run one agent step: API call + tool execution.

    Returns (step_entry, should_stop). Mutates messages in-place and updates totals dict.
    """
    t_api = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=NAIVE_TOOLS, temperature=0.0,
        )
    except Exception as e:
        print(f"  API ERROR: {e}")
        return None, True
    api_time = time.monotonic() - t_api

    msg = response.choices[0].message
    in_tok, out_tok, cached_tok, managed_tok = extract_usage(response)
    if totals["instruction_tokens"] is None:
        totals["instruction_tokens"] = in_tok
    task_ctx = max(0, in_tok - totals["instruction_tokens"])
    totals["total_in"] += in_tok
    totals["total_out"] += out_tok
    totals["total_cached"] += cached_tok
    totals["total_managed"] += managed_tok

    print(f"  tokens: {in_tok:,} in / {out_tok:,} out  api: {api_time:.1f}s")

    step_entry = make_step_entry(
        step_num, in_tok, out_tok, cached_tok, managed_tok, task_ctx,
        totals["total_in"], totals["total_out"], totals["total_managed"],
        msg.content or "", len(messages), api_time, t0, task,
    )

    if not msg.tool_calls:
        content = msg.content or ""
        print(f"  TEXT: {content[:150]}")
        messages.append({"role": "assistant", "content": content})
        totals["stalls"] += 1
        step_entry["step_time_ms"] = round(api_time * 1000)

        if totals["stalls"] >= 3:
            print("  STALL LIMIT -- stopping")
            return step_entry, True

        if any(phrase in content.lower() for phrase in DONE_PHRASES):
            print("  Model declares done")
            return step_entry, True

        messages.append({"role": "user", "content": "Use your tools to make progress."})
        return step_entry, False

    totals["stalls"] = 0
    tools_detail, tool_time = process_tool_calls(msg, executor, messages)
    step_entry["tools"] = tools_detail
    step_entry["tool_time_ms"] = round(tool_time * 1000)
    step_entry["step_time_ms"] = round((api_time + tool_time) * 1000)
    return step_entry, False


@register_strategy("window")
class WindowStrategy:
    """Sliding window: keep sys + user + last window_k messages."""

    name = "window"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        window_k: int = 20,
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

        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(messages)}")

            # Apply sliding window: keep sys + user + last window_k messages
            if len(messages) > 2 + window_k:
                dropped = len(messages) - 2 - window_k
                messages = messages[:2] + messages[2 + dropped:]
                print(f"  (window: dropped {dropped} old messages, keeping {window_k})")

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
