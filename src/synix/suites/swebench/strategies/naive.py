"""Naive strategy: full message history, no context management."""

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


@register_strategy("naive")
class NaiveStrategy:
    """Full message history, no context management."""

    name = "naive"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        **kwargs,
    ) -> dict:
        messages = [
            {"role": "system", "content": NAIVE_SYSTEM},
            {"role": "user", "content": task},
        ]

        trace = []
        total_in = total_out = total_cached = total_managed = 0
        instruction_tokens = None
        stalls = 0
        t0 = time.monotonic()

        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(messages)}")

            t_api = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, tools=NAIVE_TOOLS, temperature=0.0,
                )
            except Exception as e:
                print(f"  API ERROR: {e}")
                break
            api_time = time.monotonic() - t_api

            msg = response.choices[0].message
            in_tok, out_tok, cached_tok, managed_tok = extract_usage(response)
            if instruction_tokens is None:
                instruction_tokens = in_tok
            task_ctx = max(0, in_tok - instruction_tokens)
            total_in += in_tok
            total_out += out_tok
            total_cached += cached_tok
            total_managed += managed_tok

            print(f"  tokens: {in_tok:,} in / {out_tok:,} out  api: {api_time:.1f}s")

            step_entry = make_step_entry(
                step_num, in_tok, out_tok, cached_tok, managed_tok, task_ctx,
                total_in, total_out, total_managed, msg.content or "",
                len(messages), api_time, t0, task,
            )

            if not msg.tool_calls:
                content = msg.content or ""
                print(f"  TEXT: {content[:150]}")
                messages.append({"role": "assistant", "content": content})
                stalls += 1
                step_entry["step_time_ms"] = round(api_time * 1000)
                trace.append(step_entry)

                if stalls >= 3:
                    print("  STALL LIMIT -- stopping")
                    break

                if any(phrase in content.lower() for phrase in DONE_PHRASES):
                    print("  Model declares done")
                    break

                messages.append({"role": "user", "content": "Use your tools to make progress."})
                continue

            stalls = 0
            tools_detail, tool_time = process_tool_calls(msg, executor, messages)
            step_entry["tools"] = tools_detail
            step_entry["tool_time_ms"] = round(tool_time * 1000)
            step_entry["step_time_ms"] = round((api_time + tool_time) * 1000)
            trace.append(step_entry)

        elapsed = time.monotonic() - t0
        return {
            "trace": trace,
            "total_in": total_in,
            "total_out": total_out,
            "total_cached": total_cached,
            "total_managed": total_managed,
            "instruction_tokens": instruction_tokens or 0,
            "elapsed_s": round(elapsed, 1),
        }
