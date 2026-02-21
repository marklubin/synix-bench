"""Incremental summary strategy: rolling LLM compression after each step."""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.strategies._common import (
    DONE_PHRASES,
    extract_usage,
    make_step_entry,
)
from synix.suites.swebench.tools import NAIVE_SYSTEM, NAIVE_TOOLS

log = logging.getLogger(__name__)


@register_strategy("incremental_summary")
class IncrementalSummaryStrategy:
    """Rolling LLM compression: update a running summary after each step."""

    name = "incremental_summary"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        summary_max_tokens: int = 500,
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
        running_summary = ""
        summary_calls = 0
        t0 = time.monotonic()

        full_history = list(messages)

        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(full_history)} (full), "
                  f"summary={len(running_summary)} chars")

            # Summarize after step 1 (from step 2 onward)
            summary_tokens_in = 0
            summary_tokens_out = 0

            if step_num >= 2 and len(full_history) > 2:
                if running_summary == "":
                    latest_parts = []
                    for m in full_history[2:]:
                        role = m.get("role", "unknown")
                        content = m.get("content", "") or ""
                        if m.get("tool_calls"):
                            for tc in m["tool_calls"]:
                                content += f"\n[tool_call: {tc['function']['name']}]"
                        latest_parts.append(f"[{role}] {content[:1000]}")
                else:
                    latest_parts = []
                    last_asst_idx = None
                    for i in range(len(full_history) - 1, 1, -1):
                        if full_history[i].get("role") == "assistant":
                            last_asst_idx = i
                            break
                    if last_asst_idx is not None:
                        for m in full_history[last_asst_idx:]:
                            role = m.get("role", "unknown")
                            content = m.get("content", "") or ""
                            if m.get("tool_calls"):
                                for tc in m["tool_calls"]:
                                    content += f"\n[tool_call: {tc['function']['name']}]"
                            latest_parts.append(f"[{role}] {content[:1000]}")

                if latest_parts:
                    latest_exchange_text = "\n---\n".join(latest_parts)
                    try:
                        sum_resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": (
                                    "You are a concise summarizer. Update the running summary "
                                    "with the latest exchange. Preserve file paths, code changes, "
                                    "test results, and key decisions. Output ONLY the updated summary."
                                )},
                                {"role": "user", "content": (
                                    f"CURRENT SUMMARY:\n{running_summary}\n\n"
                                    f"LATEST EXCHANGE:\n{latest_exchange_text}"
                                )},
                            ],
                            max_tokens=summary_max_tokens,
                            temperature=0.0,
                        )
                        running_summary = sum_resp.choices[0].message.content or ""
                        s_usage = sum_resp.usage
                        summary_tokens_in = s_usage.prompt_tokens if s_usage else 0
                        summary_tokens_out = s_usage.completion_tokens if s_usage else 0
                        total_in += summary_tokens_in
                        total_out += summary_tokens_out
                        summary_calls += 1
                        print(f"  Summary updated: {len(running_summary)} chars, "
                              f"{summary_tokens_in} in / {summary_tokens_out} out")
                    except Exception as e:
                        log.error("Incremental summary failed at step %d: %s", step_num, e)

            # Reconstruct the view the model sees
            view_messages = [messages[0], messages[1]]
            if running_summary:
                view_messages.append({
                    "role": "assistant",
                    "content": f"[Running summary of previous work]\n{running_summary}",
                })
            if len(full_history) > 2:
                last_asst_idx = None
                for i in range(len(full_history) - 1, 1, -1):
                    if full_history[i].get("role") == "assistant":
                        last_asst_idx = i
                        break
                if last_asst_idx is not None:
                    for m in full_history[last_asst_idx:]:
                        view_messages.append(m)

            # API call with reconstructed view
            t_api = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=model, messages=view_messages, tools=NAIVE_TOOLS, temperature=0.0,
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
                len(view_messages), api_time, t0, task,
                summary_length_chars=len(running_summary),
                summary_calls=summary_calls,
                summary_tokens_in=summary_tokens_in,
                summary_tokens_out=summary_tokens_out,
            )

            if not msg.tool_calls:
                content = msg.content or ""
                print(f"  TEXT: {content[:150]}")
                full_history.append({"role": "assistant", "content": content})
                stalls += 1
                step_entry["step_time_ms"] = round(api_time * 1000)
                trace.append(step_entry)

                if stalls >= 3:
                    print("  STALL LIMIT -- stopping")
                    break

                if any(phrase in content.lower() for phrase in DONE_PHRASES):
                    print("  Model declares done")
                    break

                full_history.append({"role": "user", "content": "Use your tools to make progress."})
                continue

            stalls = 0
            content = msg.content or ""
            assistant_msg: dict = {"role": "assistant", "content": content}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            full_history.append(assistant_msg)

            t_tools = time.monotonic()
            tools_detail = []
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    tc_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tc_args = {}

                result_text = executor(name, tc_args)
                short = result_text[:120].replace("\n", "\\n")
                print(f"  {name}: {short}")

                sanitized = dict(tc_args)
                if name in ("write_file",) and "content" in sanitized:
                    c = sanitized["content"]
                    sanitized["content"] = c[:300] + f"...({len(c)} chars)" if len(c) > 300 else c

                tools_detail.append({"name": name, "args": sanitized, "result": result_text[:2000]})
                full_history.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

            tool_time = time.monotonic() - t_tools
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
