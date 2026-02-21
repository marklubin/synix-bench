"""RAG strategy: BM25 retrieval from memory bank + recent window."""

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
    retrieve_from_bank,
)
from synix.suites.swebench.tools import NAIVE_SYSTEM, NAIVE_TOOLS


@register_strategy("rag")
class RAGStrategy:
    """Retrieval-augmented context: BM25 from memory bank + recent window."""

    name = "rag"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        rag_window: int = 6,
        rag_top_k: int = 5,
        **kwargs,
    ) -> dict:
        messages: list[dict] = [
            {"role": "system", "content": NAIVE_SYSTEM},
            {"role": "user", "content": task},
        ]

        memory_bank: list[dict] = []
        trace: list[dict] = []
        total_in = total_out = total_cached = total_managed = 0
        instruction_tokens = None
        stalls = 0
        t0 = time.monotonic()

        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*60}")
            print(f"[Step {step_num}] messages={len(messages)}  bank={len(memory_bank)}")

            # Build reconstructed messages for the API call
            prefix = messages[:2]
            conversation_tail = messages[2:]
            if len(conversation_tail) > rag_window:
                tail = conversation_tail[-rag_window:]
            else:
                tail = conversation_tail

            # Build query from the tail for retrieval
            query_parts: list[str] = []
            for m in tail:
                content = m.get("content") or ""
                if content:
                    query_parts.append(content[:500])
                for tc in m.get("tool_calls", []):
                    func = tc.get("function", {})
                    query_parts.append(func.get("name", ""))
                    query_parts.append(func.get("arguments", "")[:200])
            query_text = " ".join(query_parts)

            retrieved_entries = retrieve_from_bank(memory_bank, query_text, rag_top_k)
            rag_retrieved = len(retrieved_entries)

            # Collect retrieved messages, skip entries whose step overlaps tail
            tail_start_step = step_num - (len(tail) // 2)
            retrieved_messages: list[dict] = []
            for entry in retrieved_entries:
                if entry["step"] >= tail_start_step:
                    continue
                retrieved_messages.extend(entry["messages"])

            reconstructed_messages = prefix + retrieved_messages + tail

            if rag_retrieved:
                print(f"  RAG: retrieved {rag_retrieved} exchanges, "
                      f"injected {len(retrieved_messages)} msgs, "
                      f"window={len(tail)}, "
                      f"total_to_model={len(reconstructed_messages)}")

            # API call
            t_api = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=model, messages=reconstructed_messages,
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
                rag_retrieved=rag_retrieved,
                rag_bank_size=len(memory_bank),
            )

            if not msg.tool_calls:
                content = msg.content or ""
                print(f"  TEXT: {content[:150]}")
                messages.append({"role": "assistant", "content": content})
                stalls += 1
                step_entry["step_time_ms"] = round(api_time * 1000)
                trace.append(step_entry)

                memory_bank.append({
                    "text": content[:1000],
                    "messages": [{"role": "assistant", "content": content}],
                    "step": step_num,
                })

                if stalls >= 3:
                    print("  STALL LIMIT -- stopping")
                    break

                if any(phrase in content.lower() for phrase in DONE_PHRASES):
                    print("  Model declares done")
                    break

                messages.append({"role": "user", "content": "Use your tools to make progress."})
                continue

            stalls = 0
            content = msg.content or ""
            assistant_msg: dict = {"role": "assistant", "content": content}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            messages.append(assistant_msg)

            exchange_messages: list[dict] = [assistant_msg]
            exchange_text_parts: list[str] = [content[:500]]

            t_tools = time.monotonic()
            tools_detail: list[dict] = []
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

                tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                messages.append(tool_msg)
                exchange_messages.append(tool_msg)
                exchange_text_parts.append(f"{name} {json.dumps(tc_args)[:200]} {result_text[:500]}")

            tool_time = time.monotonic() - t_tools
            step_entry["tools"] = tools_detail
            step_entry["tool_time_ms"] = round(tool_time * 1000)
            step_entry["step_time_ms"] = round((api_time + tool_time) * 1000)
            trace.append(step_entry)

            memory_bank.append({
                "text": " ".join(exchange_text_parts),
                "messages": exchange_messages,
                "step": step_num,
            })

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
