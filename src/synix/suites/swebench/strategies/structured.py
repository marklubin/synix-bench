"""Structured summary strategy: JSON-schema-enforced periodic summarization."""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy
from synix.suites.swebench.strategies._common import (
    DONE_PHRASES,
    STRUCTURED_REQUIRED_KEYS,
    STRUCTURED_SUMMARY_SYSTEM,
    extract_usage,
    make_step_entry,
    process_tool_calls,
)
from synix.suites.swebench.tools import NAIVE_SYSTEM, NAIVE_TOOLS

log = logging.getLogger(__name__)


@register_strategy("structured_summary")
class StructuredSummaryStrategy:
    """JSON-schema-enforced periodic summarization."""

    name = "structured_summary"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        structured_interval: int = 5,
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
        structured_summary_calls = 0
        t0 = time.monotonic()

        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(messages)}")

            # Structured summarization at intervals
            summary_tokens_in = 0
            summary_tokens_out = 0
            tracked_files = 0
            tracked_errors = 0

            if step_num > 1 and (step_num - 1) % structured_interval == 0 and len(messages) > 5:
                print(f"  (structured summarization at step {step_num})")

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
                    sum_resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": STRUCTURED_SUMMARY_SYSTEM},
                            {"role": "user", "content": summary_text_input},
                        ],
                        temperature=0.0,
                    )
                    raw_summary = sum_resp.choices[0].message.content or ""
                    s_usage = sum_resp.usage
                    summary_tokens_in = s_usage.prompt_tokens if s_usage else 0
                    summary_tokens_out = s_usage.completion_tokens if s_usage else 0
                    total_in += summary_tokens_in
                    total_out += summary_tokens_out
                    structured_summary_calls += 1

                    parsed = None
                    try:
                        clean = raw_summary.strip()
                        if clean.startswith("```"):
                            lines = clean.split("\n")
                            clean = "\n".join(
                                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                            )
                        parsed = json.loads(clean)
                    except json.JSONDecodeError as je:
                        log.warning(
                            "Structured summary JSON parse failed at step %d: %s. "
                            "Using raw text as fallback.", step_num, je,
                        )

                    if parsed is not None:
                        missing_keys = STRUCTURED_REQUIRED_KEYS - set(parsed.keys())
                        if missing_keys:
                            log.warning(
                                "Structured summary missing keys %s at step %d. "
                                "Filling with defaults.", missing_keys, step_num,
                            )
                            for k in missing_keys:
                                parsed[k] = "" if k == "current_plan" else []

                        tracked_files = len(parsed.get("files_modified", []))
                        tracked_errors = len(parsed.get("errors_encountered", []))

                        files_str = ", ".join(parsed.get("files_modified", [])) or "(none)"
                        decisions_str = "; ".join(parsed.get("decisions_made", [])) or "(none)"
                        errors_str = "; ".join(parsed.get("errors_encountered", [])) or "(none)"
                        plan_str = parsed.get("current_plan", "") or "(none)"
                        facts_str = "; ".join(parsed.get("key_facts", [])) or "(none)"

                        formatted = (
                            f"[Structured Summary]\n"
                            f"Files modified: {files_str}\n"
                            f"Decisions: {decisions_str}\n"
                            f"Errors: {errors_str}\n"
                            f"Current plan: {plan_str}\n"
                            f"Key facts: {facts_str}"
                        )
                    else:
                        formatted = f"[Structured Summary]\n{raw_summary}"

                    last_3 = messages[-3:]
                    messages = messages[:2] + [
                        {"role": "assistant", "content": formatted}
                    ] + last_3
                    print(f"  Structured summary: {len(formatted)} chars, "
                          f"{summary_tokens_in} in / {summary_tokens_out} out, "
                          f"{tracked_files} files, {tracked_errors} errors")

                except Exception as e:
                    log.error("Structured summarization failed at step %d: %s", step_num, e)

            # Agent step
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
                tracked_files=tracked_files,
                tracked_errors=tracked_errors,
                structured_summary_calls=structured_summary_calls,
            )

            if summary_tokens_in or summary_tokens_out:
                step_entry["summary_tokens_in"] = summary_tokens_in
                step_entry["summary_tokens_out"] = summary_tokens_out

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
