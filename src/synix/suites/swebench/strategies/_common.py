"""Shared constants and helpers for SWE-bench strategies."""

from __future__ import annotations

import json
import logging
import re
import time

log = logging.getLogger(__name__)

MASKING_PLACEHOLDER = "[Previous output omitted for brevity]"

SUMMARY_PROMPT = (
    "Summarize the key findings, changes made, and current state from the "
    "following conversation. Be concise but preserve critical details like "
    "file paths, code changes, and test results."
)

STRUCTURED_SUMMARY_SYSTEM = (
    "Summarize the conversation into a JSON object with these exact keys:\n"
    "{\n"
    '  "files_modified": ["list of file paths modified"],\n'
    '  "decisions_made": ["key decisions and their rationale"],\n'
    '  "errors_encountered": ["errors seen and how they were resolved"],\n'
    '  "current_plan": "what the agent is currently trying to do",\n'
    '  "key_facts": ["important facts discovered (test commands, file locations, etc.)"]\n'
    "}\n"
    "Output ONLY valid JSON, no markdown fencing."
)

STRUCTURED_REQUIRED_KEYS = {
    "files_modified", "decisions_made", "errors_encountered",
    "current_plan", "key_facts",
}

DONE_PHRASES = [
    "all tests pass", "task is complete", "successfully completed",
    "implementation is complete", "fix is complete", "issue is fixed",
    "the fix has been", "changes are complete",
]


def parse_pytest_output(output: str) -> dict:
    """Parse pytest -v output to extract pass/fail/error counts."""
    passed = failed = errors = 0
    for line in output.split("\n"):
        if "passed" in line or "failed" in line or "error" in line:
            p = re.search(r"(\d+) passed", line)
            f = re.search(r"(\d+) failed", line)
            e = re.search(r"(\d+) error", line)
            if p:
                passed = int(p.group(1))
            if f:
                failed = int(f.group(1))
            if e:
                errors = int(e.group(1))
    total = passed + failed + errors
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "all_passed": total > 0 and failed == 0 and errors == 0,
    }


def mask_old_tool_outputs(messages: list[dict], window: int = 10) -> list[dict]:
    """Replace tool result content older than `window` turns from the end.

    Follows The Complexity Trap (NeurIPS 2025 Workshop) methodology:
    - Only masks role="tool" message content
    - Preserves assistant reasoning text and tool call structure
    - Keeps the most recent `window` messages unmasked
    """
    if len(messages) <= window:
        return messages

    cutoff = len(messages) - window
    masked = []
    for i, msg in enumerate(messages):
        if i < cutoff and msg.get("role") == "tool":
            masked.append({**msg, "content": MASKING_PLACEHOLDER})
        else:
            masked.append(msg)
    return masked


def make_step_entry(
    step_num: int,
    in_tok: int,
    out_tok: int,
    cached_tok: int,
    managed_tok: int,
    task_ctx: int,
    total_in: int,
    total_out: int,
    total_managed: int,
    content: str,
    messages_len: int,
    api_time: float,
    t0: float,
    task: str,
    **extra: object,
) -> dict:
    """Build a standardized step trace entry for non-stack strategies."""
    return {
        "step": step_num, "depth": 0,
        "input_tokens": in_tok, "output_tokens": out_tok,
        "cached_tokens": cached_tok, "managed_tokens": managed_tok,
        "task_context_tokens": task_ctx,
        "cumulative_input": total_in, "cumulative_output": total_out,
        "cumulative_managed": total_managed,
        "content": content, "tools": [],
        "event": None,
        "stack": [{"depth": 0, "name": "main", "objective": task[:80], "frame_steps": step_num}],
        "frame_messages": messages_len, "context": [],
        "api_time_ms": round(api_time * 1000), "tool_time_ms": 0, "step_time_ms": 0,
        "wall_clock_s": round(time.monotonic() - t0, 2),
        "heap": {}, "heap_total_chars": 0, "heap_events": [], "heap_freed": [],
        "registers": {}, "reg_bytes_total": 0, "objective": task[:80], "frame_steps": step_num,
        **extra,
    }


def make_result(trace: list[dict], totals: dict, elapsed: float) -> dict:
    """Build a standardized result dict from a strategy run."""
    return {
        "trace": trace,
        "total_in": totals["total_in"],
        "total_out": totals["total_out"],
        "total_cached": totals["total_cached"],
        "total_managed": totals["total_managed"],
        "instruction_tokens": totals["instruction_tokens"] or 0,
        "elapsed_s": round(elapsed, 1),
    }


def extract_usage(response) -> tuple[int, int, int, int]:
    """Extract token counts from an OpenAI response.

    Returns (in_tok, out_tok, cached_tok, managed_tok).
    """
    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    cached_tok = 0
    if usage and usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        cached_tok = usage.prompt_tokens_details.cached_tokens
    managed_tok = in_tok - cached_tok
    return in_tok, out_tok, cached_tok, managed_tok


def process_tool_calls(msg, executor, messages: list[dict]) -> tuple[list[dict], float]:
    """Process tool calls from a model response.

    Appends the assistant message and tool results to `messages` (mutates in-place).
    Returns (tools_detail, tool_time_seconds).
    """
    content = msg.content or ""
    assistant_msg: dict = {"role": "assistant", "content": content}
    assistant_msg["tool_calls"] = [
        {"id": tc.id, "type": "function",
         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in msg.tool_calls
    ]
    messages.append(assistant_msg)

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
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    tool_time = time.monotonic() - t_tools
    return tools_detail, tool_time


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Simple BM25 scoring for a single document."""
    dl = len(doc_tokens)
    score = 0.0
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    for qt in set(query_tokens):
        tf = tf_map.get(qt, 0)
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
            score += numerator / denominator
    return score


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer for BM25."""
    return text.lower().split()


def retrieve_from_bank(
    memory_bank: list[dict],
    query_text: str,
    top_k: int,
) -> list[dict]:
    """Retrieve top-k entries from memory_bank using BM25 scoring."""
    if not memory_bank or not query_text.strip():
        return []

    query_tokens = _bm25_tokenize(query_text)
    if not query_tokens:
        return []

    doc_token_lists = [_bm25_tokenize(entry["text"]) for entry in memory_bank]
    avg_dl = sum(len(dt) for dt in doc_token_lists) / max(len(doc_token_lists), 1)

    scored: list[tuple[float, int]] = []
    for idx, doc_tokens in enumerate(doc_token_lists):
        s = _bm25_score(query_tokens, doc_tokens, avg_dl)
        if s > 0:
            scored.append((s, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [memory_bank[idx] for _, idx in scored[:top_k]]
