"""Summary strategy: LLM summarization every N steps."""

from __future__ import annotations

import logging
import time

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.strategies._common import SUMMARY_PROMPT
from synix.suites.swebench.strategies.window import _run_agent_step
from synix.suites.swebench.tools import NAIVE_SYSTEM

log = logging.getLogger(__name__)


@register_strategy("summary")
class SummaryStrategy:
    """LLM summarization: compress conversation every summary_interval steps."""

    name = "summary"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        summary_interval: int = 5,
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

            # Every summary_interval steps, compress the conversation
            summary_tokens_in = 0
            summary_tokens_out = 0
            if step_num > 1 and (step_num - 1) % summary_interval == 0 and len(messages) > 5:
                print(f"  (summarizing conversation at step {step_num})")
                to_summarize = messages[2:-3]
                summary_input = []
                for m in to_summarize:
                    role = m.get("role", "unknown")
                    content = m.get("content", "") or ""
                    if m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            content += f"\n[tool_call: {tc['function']['name']}]"
                    summary_input.append(f"[{role}] {content[:1000]}")

                summary_text_input = "\n---\n".join(summary_input)
                try:
                    summary_response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SUMMARY_PROMPT},
                            {"role": "user", "content": summary_text_input},
                        ],
                        temperature=0.0,
                    )
                    summary_content = summary_response.choices[0].message.content or ""
                    s_usage = summary_response.usage
                    summary_tokens_in = s_usage.prompt_tokens if s_usage else 0
                    summary_tokens_out = s_usage.completion_tokens if s_usage else 0
                    totals["total_in"] += summary_tokens_in
                    totals["total_out"] += summary_tokens_out

                    last_3 = messages[-3:]
                    messages = messages[:2] + [
                        {"role": "assistant", "content": f"[Summary of previous work]\n{summary_content}"}
                    ] + last_3
                    print(f"  Summary: {len(summary_content)} chars, "
                          f"{summary_tokens_in} in / {summary_tokens_out} out")
                except Exception as e:
                    log.error("Summarization failed at step %d: %s", step_num, e)

            step_entry, should_stop = _run_agent_step(
                client, model, messages, executor, step_num, t0, totals, task,
            )
            if step_entry is None:
                break
            if summary_tokens_in or summary_tokens_out:
                step_entry["summary_tokens_in"] = summary_tokens_in
                step_entry["summary_tokens_out"] = summary_tokens_out
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
