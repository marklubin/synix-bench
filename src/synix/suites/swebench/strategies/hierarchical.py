"""Hierarchical strategy: 3-tier context (hot/warm/cold)."""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.strategies._common import (
    DONE_PHRASES,
    MASKING_PLACEHOLDER,
    extract_usage,
    make_step_entry,
    process_tool_calls,
)
from synix.suites.swebench.tools import NAIVE_SYSTEM, NAIVE_TOOLS

log = logging.getLogger(__name__)


@register_strategy("hierarchical")
class HierarchicalStrategy:
    """3-tier hierarchical context: hot (full), warm (masked), cold (summary)."""

    name = "hierarchical"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        hot_window: int = 5,
        warm_window: int = 20,
        cold_interval: int = 10,
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
        cold_summary = ""
        last_cold_end = 0
        t0 = time.monotonic()

        for step_num in range(1, max_steps + 1):
            step_t0 = time.monotonic()
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(messages)}")

            # Tier computation
            conversation = messages[2:]
            n = len(conversation)

            hot_start = max(0, n - hot_window)
            warm_start = max(0, hot_start - warm_window)

            hot_msgs = conversation[hot_start:]
            warm_msgs = conversation[warm_start:hot_start]

            # Mask warm-tier tool outputs
            warm_masked = []
            warm_masked_count = 0
            for m in warm_msgs:
                if m.get("role") == "tool":
                    warm_masked.append({**m, "content": MASKING_PLACEHOLDER})
                    warm_masked_count += 1
                else:
                    warm_masked.append(m)

            # Build reconstructed message list
            reconstructed = [messages[0], messages[1]]
            if cold_summary:
                reconstructed.append(
                    {"role": "assistant", "content": f"[Summary of earlier work]\n{cold_summary}"}
                )
            reconstructed.extend(warm_masked)
            reconstructed.extend(hot_msgs)

            tier_hot = len(hot_msgs)
            tier_warm = len(warm_msgs)

            print(
                f"  tiers: hot={tier_hot} warm={tier_warm} "
                f"(masked={warm_masked_count}) cold_chars={len(cold_summary)}"
            )

            # Cold summarization (every cold_interval steps)
            cold_summary_tokens_in = 0
            cold_summary_tokens_out = 0

            if step_num > 1 and step_num % cold_interval == 0:
                cold_zone = conversation[:warm_start]
                if warm_start > last_cold_end and cold_zone:
                    new_exchanges = cold_zone[last_cold_end:]
                    new_exchanges_parts = []
                    for m in new_exchanges:
                        role = m.get("role", "unknown")
                        content = m.get("content", "") or ""
                        if m.get("tool_calls"):
                            for tc in m["tool_calls"]:
                                content += f"\n[tool_call: {tc['function']['name']}]"
                        new_exchanges_parts.append(f"[{role}] {content[:1000]}")
                    new_exchanges_text = "\n---\n".join(new_exchanges_parts)

                    print(
                        f"  (cold summarization: {len(new_exchanges)} new messages, "
                        f"cold_zone total={len(cold_zone)})"
                    )
                    try:
                        summary_response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": (
                                    "You are a concise summarizer. Produce an updated summary "
                                    "incorporating the new exchanges. Preserve file paths, code "
                                    "changes, test results, and key decisions."
                                )},
                                {"role": "user", "content": (
                                    f"PREVIOUS SUMMARY:\n{cold_summary}\n\n"
                                    f"NEW EXCHANGES:\n{new_exchanges_text}"
                                )},
                            ],
                            temperature=0.0,
                        )
                        cold_summary = summary_response.choices[0].message.content or ""
                        s_usage = summary_response.usage
                        cold_summary_tokens_in = s_usage.prompt_tokens if s_usage else 0
                        cold_summary_tokens_out = s_usage.completion_tokens if s_usage else 0
                        total_in += cold_summary_tokens_in
                        total_out += cold_summary_tokens_out
                        last_cold_end = warm_start
                        print(
                            f"  Cold summary updated: {len(cold_summary)} chars, "
                            f"{cold_summary_tokens_in} in / {cold_summary_tokens_out} out"
                        )
                    except Exception as e:
                        log.error("Cold summarization FAILED at step %d: %s", step_num, e)

            # Main API call (uses reconstructed, not messages)
            t_api = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=model, messages=reconstructed,
                    tools=NAIVE_TOOLS, temperature=0.0,
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
                tier_hot=tier_hot,
                tier_warm=tier_warm,
                tier_warm_masked=warm_masked_count,
                tier_cold_summary_chars=len(cold_summary),
                cold_summary_tokens_in=cold_summary_tokens_in,
                cold_summary_tokens_out=cold_summary_tokens_out,
            )

            if not msg.tool_calls:
                content = msg.content or ""
                print(f"  TEXT: {content[:150]}")
                messages.append({"role": "assistant", "content": content})
                stalls += 1
                step_entry["step_time_ms"] = round((time.monotonic() - step_t0) * 1000)
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
            step_entry["step_time_ms"] = round((time.monotonic() - step_t0) * 1000)
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
