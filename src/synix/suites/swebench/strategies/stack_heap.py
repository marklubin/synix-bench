"""Stack+Heap strategy: agent-controlled context via push/pop + heap memory.

This wraps the StackAgent class from HMB's stack_prototype.py. The core
agent is self-contained here; tool schemas, frame/heap data structures,
and the full step() loop are all included.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from synix.debug.probes import Probe
from synix.executor.base import ToolExecutor
from synix.suites.swebench.strategies.base import register_strategy


# ── Layout config ────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).resolve().parents[4] / "prompts"
LAYOUTS_DIR = Path(__file__).resolve().parents[4] / "configs" / "layouts"

DEFAULT_LAYOUT = {
    "name": "baseline",
    "prompt_file": "stack-heap-v7.txt",
    "sections": ["system_prompt", "registers", "heap_index", "objective", "conversation"],
    "masking": {"enabled": False, "window": 10, "placeholder": "[Output omitted]"},
    "registers": {
        "R0": {"label": "INTENT", "max_chars": 200},
        "R1": {"label": "RETURN", "max_chars": 200},
        "R3": {"label": "BUDGET", "max_chars": 100},
    },
    "push_limits": {"objective": 200, "context": 500, "return_spec": 200},
}


def load_layout(path: str | Path) -> dict:
    """Load a layout config JSON. Returns dict with all fields populated."""
    with open(path) as f:
        layout = json.load(f)
    merged = {**DEFAULT_LAYOUT, **layout}
    merged["registers"] = {**DEFAULT_LAYOUT["registers"], **layout.get("registers", {})}
    merged["push_limits"] = {**DEFAULT_LAYOUT["push_limits"], **layout.get("push_limits", {})}
    merged["masking"] = {**DEFAULT_LAYOUT["masking"], **layout.get("masking", {})}
    return merged


def load_system_prompt(layout: dict) -> str:
    """Load and patch the system prompt with layout-specific sizes."""
    prompt_file = layout.get("prompt_file", "stack-heap-v7.txt")
    prompt_path = PROMPTS_DIR / prompt_file
    if not prompt_path.exists():
        return f"You are a skilled software engineer. Use push_frame/pop_frame to organize work."
    text = prompt_path.read_text()
    regs = layout.get("registers", DEFAULT_LAYOUT["registers"])
    push = layout.get("push_limits", DEFAULT_LAYOUT["push_limits"])
    replacements = [
        ("| R0  | INTENT | 200 chars |", f"| R0  | INTENT | {regs['R0']['max_chars']} chars |"),
        ("| R1  | RETURN | 200 chars |", f"| R1  | RETURN | {regs['R1']['max_chars']} chars |"),
        ("| R3  | BUDGET | 100 chars |", f"| R3  | BUDGET | {regs['R3']['max_chars']} chars |"),
        ("(max 200 chars): What to accomplish", f"(max {push['objective']} chars): What to accomplish"),
        ("(max 500 chars): File paths", f"(max {push['context']} chars): File paths"),
        ("(max 200 chars): What to report", f"(max {push['return_spec']} chars): What to report"),
        ("This way 200 chars of register", f"This way {regs['R0']['max_chars']} chars of register"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ── Data structures ──────────────────────────────────────────────────


def _empty_registers(schema: dict) -> dict[str, str]:
    return {k: "" for k in schema}


def _init_root_registers(task: str, return_spec: str, schema: dict) -> dict[str, str]:
    regs = _empty_registers(schema)
    regs["R0"] = _truncate_register("R0", task, schema)
    regs["R1"] = _truncate_register("R1", return_spec, schema)
    return regs


def _init_child_registers(
    objective: str, return_spec: str, schema: dict, budget_hint: str = ""
) -> dict[str, str]:
    regs = _empty_registers(schema)
    regs["R0"] = _truncate_register("R0", objective, schema)
    regs["R1"] = _truncate_register("R1", return_spec, schema)
    if budget_hint:
        regs["R3"] = _truncate_register("R3", budget_hint, schema)
    return regs


def _truncate_register(key: str, value: str, schema: dict) -> str:
    max_c = schema[key]["max_chars"]
    if len(value) <= max_c:
        return value
    return value[:max_c - 12] + " [TRUNCATED]"


def _reg_bytes_total(regs: dict[str, str]) -> int:
    return sum(len(v) for v in regs.values())


@dataclass
class Frame:
    """One stack frame -- a focused sub-task with its own conversation."""
    name: str
    objective: str
    context: str
    return_spec: str
    messages: list[dict] = field(default_factory=list)
    depth: int = 0
    steps: int = 0
    registers: dict[str, str] = field(
        default_factory=lambda: _empty_registers(DEFAULT_LAYOUT["registers"])
    )


@dataclass
class HeapChunk:
    """A named persistent memory chunk -- lives across all frames."""
    name: str
    content: str
    description: str
    allocated_at: int
    last_written_at: int
    last_read_at: int | None = None
    size_chars: int = 0


# ── Tool schemas ─────────────────────────────────────────────────────

STACK_HEAP_TOOLS = [
    {"type": "function", "function": {
        "name": "push_frame",
        "description": (
            "Start a focused sub-task with a CLEAN context -- like calling a function. "
            "Your current conversation is SAVED but becomes INVISIBLE. The sub-task "
            "starts with ONLY: system prompt + objective + context you provide here."
        ),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Short kebab-case slug for this sub-task"},
            "objective": {"type": "string", "description": "What this sub-task must accomplish"},
            "context": {"type": "string", "description": "Brief orientation (max 500 chars)."},
            "return_spec": {"type": "string", "description": "What the sub-task must return via pop_frame"},
        }, "required": ["name", "objective", "context", "return_spec"]},
    }},
    {"type": "function", "function": {
        "name": "pop_frame",
        "description": "Complete the current sub-task and return to the caller.",
        "parameters": {"type": "object", "properties": {
            "result": {"type": "string", "description": "The return value."},
        }, "required": ["result"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file (creates directories as needed)",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "content": {"type": "string", "description": "File content to write"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the workspace",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path relative to workspace (default: '.')"},
        }},
    }},
    {"type": "function", "function": {
        "name": "heap_alloc",
        "description": "Store a named persistent memory chunk.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Unique name (kebab-case)"},
            "content": {"type": "string", "description": "Content to store"},
            "description": {"type": "string", "description": "Short one-line description"},
        }, "required": ["name", "content"]},
    }},
    {"type": "function", "function": {
        "name": "heap_read",
        "description": "Load a heap chunk's content into this conversation.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Name of the heap chunk to read"},
        }, "required": ["name"]},
    }},
    {"type": "function", "function": {
        "name": "heap_write",
        "description": "Overwrite the content of an existing heap chunk.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Name of an existing heap chunk"},
            "content": {"type": "string", "description": "New content"},
        }, "required": ["name", "content"]},
    }},
    {"type": "function", "function": {
        "name": "heap_free",
        "description": "Free a heap chunk -- removes it from the prompt.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Name of the heap chunk to free"},
        }, "required": ["name"]},
    }},
]


def _sanitize_args(name: str, args: dict) -> dict:
    """Return args safe for trace (truncate write_file/heap content)."""
    if name in ("write_file", "heap_alloc", "heap_write"):
        content = args.get("content", "")
        return {
            **args,
            "content": content[:300] + f"...({len(content)} chars)" if len(content) > 300 else content,
        }
    return args


# ── StackAgent ────────────────────────────────────────────────────────


class StackAgent:
    """Stack-based context management agent.

    The agent controls its own context via push_frame/pop_frame (stack)
    and heap_alloc/heap_read/heap_write/heap_free (persistent memory).
    """

    MAX_DEPTH = 5
    MAX_TOTAL_STEPS = 50
    MAX_STALLS = 3

    def __init__(
        self,
        client: OpenAI,
        model: str,
        workspace: str,
        *,
        tool_executor: ToolExecutor | None = None,
        layout: dict | None = None,
        no_think_prefill: bool = False,
    ):
        self.client = client
        self.model = model
        self.workspace = workspace
        self.tool_executor = tool_executor
        self.layout = layout or DEFAULT_LAYOUT
        self.register_schema = self.layout["registers"]
        self.push_limits = self.layout["push_limits"]
        self.system_prompt = load_system_prompt(self.layout)
        self.HEAP_CHAR_LIMIT = self.layout.get("heap_char_limit", 10_000)
        self.no_think_prefill = no_think_prefill
        self.stack: list[Frame] = []
        self.current: Frame | None = None
        self.total_steps = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_managed_tokens = 0
        self.instruction_tokens: int | None = None
        self.trace: list[dict] = []
        self.done = False
        self.final_result: str | None = None
        self.consecutive_stalls = 0
        self.heap: dict[str, HeapChunk] = {}
        self.heap_freed: list[dict] = []
        self.peak_reg_bytes = 0
        self.probes: list[Probe] = []
        self.probe_log: list[dict] = []
        self.tools = list(STACK_HEAP_TOOLS)
        self.total_masked_outputs = 0

    def _indent(self) -> str:
        return "  " * (self.current.depth if self.current else 0)

    def _get_stack_state(self) -> list[dict]:
        state = []
        for f in self.stack:
            state.append({
                "depth": f.depth, "name": f.name,
                "objective": f.objective, "frame_steps": f.steps,
            })
        if self.current:
            state.append({
                "depth": self.current.depth, "name": self.current.name,
                "objective": self.current.objective, "frame_steps": self.current.steps,
            })
        return state

    def _snapshot_context(self, messages: list[dict]) -> list[dict]:
        snap = []
        for msg in messages:
            entry: dict = {"role": msg["role"]}
            c = msg.get("content", "") or ""
            entry["content"] = c[:1500]
            entry["content_len"] = len(c)
            if len(c) > 1500:
                entry["truncated"] = True
            if msg.get("tool_calls"):
                entry["tool_calls"] = [
                    tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                    for tc in msg["tool_calls"]
                ]
            if msg.get("tool_call_id"):
                entry["tool_call_id"] = msg["tool_call_id"]
            snap.append(entry)
        return snap

    def _build_register_block(self) -> str:
        regs = self.current.registers
        parts = ["# FRAME REGISTERS (set by parent, read-only)"]
        for key, schema in self.register_schema.items():
            label = schema["label"]
            val = regs.get(key, "")
            if val:
                parts.append(f"[{key} {label}] {val}")
        return "\n".join(parts)

    def _build_heap_block(self) -> str:
        total_chars = sum(c.size_chars for c in self.heap.values()) if self.heap else 0
        pct = round(total_chars / self.HEAP_CHAR_LIMIT * 100) if self.heap else 0
        parts = ["# HEAP STORE"]
        parts.append(f"Budget: {total_chars:,} / {self.HEAP_CHAR_LIMIT:,} chars ({pct}%)")
        parts.append("heap_read(name) loads content into this frame.\n")
        if self.heap:
            parts.append(f"{'Name':<18} {'Size':>6}  {'Created':>8}  {'LastRead':>8}  Description")
            parts.append("-" * 78)
            for name, chunk in self.heap.items():
                created = f"step {chunk.allocated_at}"
                last_read = f"step {chunk.last_read_at}" if chunk.last_read_at is not None else "never"
                desc = chunk.description[:30] if chunk.description else "(no description)"
                parts.append(f"{name:<18} {chunk.size_chars:>5}c  {created:>8}  {last_read:>8}  {desc}")
        else:
            parts.append("(empty -- no chunks stored)")
        return "\n".join(parts)

    def _mask_conversation(self, msgs: list[dict]) -> list[dict]:
        masking_cfg = self.layout.get("masking", {})
        if not masking_cfg.get("enabled", False):
            return msgs
        window = masking_cfg.get("window", 10)
        placeholder = masking_cfg.get("placeholder", "[Output omitted]")
        tool_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        if len(tool_indices) <= window:
            return msgs
        cutoff_idx = tool_indices[-window]
        masked = []
        for i, msg in enumerate(msgs):
            if i < cutoff_idx and msg.get("role") == "tool":
                masked.append({**msg, "content": placeholder})
                self.total_masked_outputs += 1
            else:
                masked.append(msg)
        return masked

    def _build_messages_with_injections(self) -> list[dict]:
        stored = self.current.messages
        system_prompt_msg = stored[0]
        objective_msg = stored[1]
        conversation_msgs = stored[2:]
        reg_msg = {"role": "system", "content": self._build_register_block()}
        heap_msg = {"role": "system", "content": self._build_heap_block()}
        conversation_msgs = self._mask_conversation(conversation_msgs)
        section_map = {
            "system_prompt": [system_prompt_msg],
            "registers": [reg_msg],
            "heap_index": [heap_msg],
            "objective": [objective_msg],
            "conversation": conversation_msgs,
        }
        result = []
        for section_id in self.layout["sections"]:
            result.extend(section_map.get(section_id, []))
        return result

    def _get_heap_snapshot(self) -> dict:
        return {
            name: {
                "size": chunk.size_chars,
                "description": chunk.description,
                "allocated_at": chunk.allocated_at,
                "last_written_at": chunk.last_written_at,
                "last_read_at": chunk.last_read_at,
            }
            for name, chunk in self.heap.items()
        }

    def _check_probes(self, event: dict | None, heap_events: list[dict], step_num: int) -> list[Probe]:
        to_fire = []
        event_type = event.get("type") if event else None
        for probe in self.probes:
            t = probe.trigger
            if t == "pop_frame" and event_type == "pop":
                to_fire.append(probe)
            elif t == "push_frame" and event_type == "push":
                to_fire.append(probe)
            elif t == "heap_alloc" and any(he["type"] == "alloc" for he in heap_events):
                to_fire.append(probe)
            elif t.startswith("step="):
                try:
                    target = int(t.split("=", 1)[1])
                except ValueError:
                    continue
                if step_num == target and not probe.fired:
                    probe.fired = True
                    to_fire.append(probe)
        return to_fire

    def _run_probe(self, question: str) -> dict:
        ind = self._indent()
        print(f"{ind}  PROBE: {question[:120]}")
        probe_msg = {
            "role": "user",
            "content": (
                "[PROBE -- researcher question, not part of your task. "
                "Answer briefly and honestly.]\n\n" + question
            ),
        }
        self.current.messages.append(probe_msg)
        messages = self._build_messages_with_injections()
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0,
            )
            answer = response.choices[0].message.content or ""
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
        except Exception as e:
            answer = f"(probe error: {e})"
            in_tok, out_tok = 0, 0
            print(f"{ind}  PROBE ERROR: {e}")
        print(f"{ind}  ANSWER: {answer[:300]}")
        self.current.messages.pop()
        result = {
            "after_step": self.total_steps, "depth": self.current.depth,
            "trigger": "manual", "question": question, "answer": answer,
            "probe_input_tokens": in_tok, "probe_output_tokens": out_tok,
        }
        self.probe_log.append(result)
        return result

    def _record_step(
        self, in_tok: int, out_tok: int, content: str, tools_detail: list[dict],
        event: dict | None, context_before: list[dict], *,
        api_time: float = 0.0, tool_time: float = 0.0,
        heap_events: list[dict] | None = None,
        cached_tokens: int = 0, managed_tokens: int = 0,
    ) -> None:
        reg_bytes = _reg_bytes_total(self.current.registers)
        self.peak_reg_bytes = max(self.peak_reg_bytes, reg_bytes)
        self.trace.append({
            "step": self.total_steps, "depth": self.current.depth,
            "objective": self.current.objective, "frame_steps": self.current.steps,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cached_tokens": cached_tokens, "managed_tokens": managed_tokens,
            "task_context_tokens": max(0, in_tok - (self.instruction_tokens or 0)),
            "cumulative_input": self.total_input_tokens,
            "cumulative_output": self.total_output_tokens,
            "cumulative_managed": self.total_managed_tokens,
            "content": content, "tools": tools_detail, "event": event,
            "stack": self._get_stack_state(),
            "frame_messages": len(self.current.messages),
            "context": context_before,
            "api_time_ms": round(api_time * 1000),
            "tool_time_ms": round(tool_time * 1000),
            "step_time_ms": round((api_time + tool_time) * 1000),
            "wall_clock": time.monotonic(),
            "heap": self._get_heap_snapshot(),
            "heap_total_chars": sum(c.size_chars for c in self.heap.values()),
            "heap_events": heap_events or [],
            "heap_freed": list(self.heap_freed),
            "registers": dict(self.current.registers),
            "reg_bytes_total": reg_bytes,
        })

    def start(self, task: str) -> None:
        return_spec = "Confirmation that all tests pass, with the test output"
        self.current = Frame(
            name="main", objective=task,
            context=f"Workspace: {self.workspace}",
            return_spec=return_spec, depth=0,
            registers=_init_root_registers(task, return_spec, self.register_schema),
        )
        self.current.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"## Task\n{task}\n\n"
                f"## Workspace\n{self.workspace}\n\n"
                "Plan your approach, then execute it. Use push_frame/pop_frame "
                "to organize distinct phases of work into focused sub-tasks."
            )},
        ]

    def step(self) -> bool:
        if self.done or self.total_steps >= self.MAX_TOTAL_STEPS:
            if not self.done:
                print(f"\n{'='*60}")
                print(f"MAX STEPS ({self.MAX_TOTAL_STEPS}) reached")
            return False

        self.total_steps += 1
        self.current.steps += 1
        ind = self._indent()

        print(f"\n{'='*60}")
        print(f"{ind}[Step {self.total_steps}] depth={self.current.depth} "
              f"frame_steps={self.current.steps} "
              f"obj='{self.current.objective[:50]}'")

        messages_for_api = self._build_messages_with_injections()
        context_before = self._snapshot_context(messages_for_api)

        try:
            t_api_start = time.monotonic()
            create_kwargs: dict[str, Any] = dict(
                model=self.model, messages=messages_for_api,
                tools=self.tools, temperature=0.0,
            )
            if self.no_think_prefill:
                create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            response = self.client.chat.completions.create(**create_kwargs)
            api_time = time.monotonic() - t_api_start
        except Exception as e:
            print(f"{ind}  API ERROR: {e}")
            self.done = True
            return False

        msg = response.choices[0].message
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cached_tok = 0
        if usage and usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
            cached_tok = usage.prompt_tokens_details.cached_tokens
        managed_tok = in_tok - cached_tok
        if self.instruction_tokens is None:
            self.instruction_tokens = in_tok
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_cached_tokens += cached_tok
        self.total_managed_tokens += managed_tok

        print(f"{ind}  tokens: {in_tok:,} in / {out_tok:,} out  api: {api_time:.1f}s")

        # No tool calls
        if not msg.tool_calls:
            content = msg.content or ""
            print(f"{ind}  TEXT: {content[:150]}")
            self.current.messages.append({"role": "assistant", "content": content})
            self.consecutive_stalls += 1
            self._record_step(
                in_tok, out_tok, content, [], None, context_before,
                api_time=api_time, cached_tokens=cached_tok, managed_tokens=managed_tok,
            )
            if self.consecutive_stalls >= self.MAX_STALLS:
                print(f"{ind}  STALL LIMIT -- auto-popping")
                return self._handle_pop(content or "(no result -- stalled)")
            self.current.messages.append({
                "role": "user",
                "content": "Use your tools to make progress. Call pop_frame when you're done.",
            })
            return True

        # Has tool calls
        self.consecutive_stalls = 0
        content = msg.content or ""
        assistant_msg: dict = {"role": "assistant", "content": content}
        assistant_msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
        self.current.messages.append(assistant_msg)

        tools_detail: list[dict] = []
        pending_push: dict | None = None
        pending_pop: dict | None = None
        event: dict | None = None
        heap_events: list[dict] = []
        t_tools_start = time.monotonic()

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name == "push_frame":
                slug = args.get("name", "unnamed")
                obj = args.get("objective", "?")
                ctx = args.get("context", "")
                ret = args.get("return_spec", "")
                OBJ_LIMIT = self.push_limits["objective"]
                CTX_LIMIT = self.push_limits["context"]
                RET_LIMIT = self.push_limits["return_spec"]
                budget_errors = []
                if len(obj) > OBJ_LIMIT:
                    budget_errors.append(f"objective too long ({len(obj)} chars, max {OBJ_LIMIT}).")
                if len(ctx) > CTX_LIMIT:
                    budget_errors.append(f"context too long ({len(ctx)} chars, max {CTX_LIMIT}).")
                if len(ret) > RET_LIMIT:
                    budget_errors.append(f"return_spec too long ({len(ret)} chars, max {RET_LIMIT}).")
                if budget_errors:
                    result_text = "Error: push_frame REJECTED.\n" + "\n".join(f"- {e}" for e in budget_errors)
                    print(f"{ind}  PUSH REJECTED: {result_text[:200]}")
                else:
                    print(f"{ind}  >>> PUSH {slug}()")
                    result_text = f"Frame pushed. Sub-task '{slug}' is starting."
                    pending_push = args
                event = {
                    "type": "push" if not budget_errors else "push_rejected",
                    "name": slug, "objective": obj, "context": ctx, "return_spec": ret,
                }

            elif name == "pop_frame":
                result_str = args.get("result", "")
                print(f"{ind}  <<< POP: {result_str[:120]}")
                result_text = f"Returning: {result_str[:200]}"
                pending_pop = args
                event = {"type": "pop", "result": result_str}

            elif name == "heap_alloc":
                chunk_name = args.get("name", "")
                chunk_content = args.get("content", "")
                chunk_desc = args.get("description", "")
                if chunk_name in self.heap:
                    result_text = f"Error: heap chunk '{chunk_name}' already exists."
                else:
                    chunk = HeapChunk(
                        name=chunk_name, content=chunk_content, description=chunk_desc,
                        allocated_at=self.total_steps, last_written_at=self.total_steps,
                        size_chars=len(chunk_content),
                    )
                    self.heap[chunk_name] = chunk
                    result_text = f"OK -- allocated '{chunk_name}' ({len(chunk_content)} chars)"
                    heap_events.append({"type": "alloc", "name": chunk_name, "size": len(chunk_content), "step": self.total_steps})

            elif name == "heap_read":
                chunk_name = args.get("name", "")
                if chunk_name not in self.heap:
                    result_text = f"Error: heap chunk '{chunk_name}' doesn't exist."
                else:
                    chunk = self.heap[chunk_name]
                    chunk.last_read_at = self.total_steps
                    result_text = chunk.content
                    heap_events.append({"type": "read", "name": chunk_name, "size": chunk.size_chars, "step": self.total_steps})

            elif name == "heap_write":
                chunk_name = args.get("name", "")
                chunk_content = args.get("content", "")
                if chunk_name not in self.heap:
                    result_text = f"Error: heap chunk '{chunk_name}' doesn't exist."
                else:
                    old_size = self.heap[chunk_name].size_chars
                    self.heap[chunk_name].content = chunk_content
                    self.heap[chunk_name].size_chars = len(chunk_content)
                    self.heap[chunk_name].last_written_at = self.total_steps
                    result_text = f"OK -- updated '{chunk_name}' ({old_size} -> {len(chunk_content)} chars)"
                    heap_events.append({"type": "write", "name": chunk_name, "size": len(chunk_content), "step": self.total_steps})

            elif name == "heap_free":
                chunk_name = args.get("name", "")
                if chunk_name not in self.heap:
                    result_text = f"Error: heap chunk '{chunk_name}' doesn't exist."
                else:
                    chunk = self.heap.pop(chunk_name)
                    self.heap_freed.append({
                        "name": chunk_name, "description": chunk.description,
                        "size": chunk.size_chars, "allocated_at": chunk.allocated_at,
                        "freed_at": self.total_steps,
                    })
                    result_text = f"OK -- freed '{chunk_name}' ({chunk.size_chars} chars recovered)"
                    heap_events.append({"type": "free", "name": chunk_name, "size": chunk.size_chars, "step": self.total_steps})

            else:
                # Regular tool -- dispatch to executor
                if self.tool_executor is not None:
                    result_text = self.tool_executor(name, args)
                else:
                    result_text = f"Error: no tool executor configured for '{name}'"
                short = result_text[:120].replace("\n", "\\n")
                print(f"{ind}  {name}: {short}")

            tools_detail.append({
                "name": name, "args": _sanitize_args(name, args),
                "result": result_text[:2000],
            })
            self.current.messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_text}
            )

        tool_time = time.monotonic() - t_tools_start
        self._record_step(
            in_tok, out_tok, content, tools_detail, event, context_before,
            api_time=api_time, tool_time=tool_time, heap_events=heap_events,
            cached_tokens=cached_tok, managed_tokens=managed_tok,
        )

        # Probes
        if self.probes:
            to_fire = self._check_probes(event, heap_events, self.total_steps)
            for probe in to_fire:
                result = self._run_probe(probe.question)
                result["trigger"] = probe.trigger

        # Frame transitions
        if pending_push is not None:
            return self._handle_push(pending_push)
        elif pending_pop is not None:
            return self._handle_pop(pending_pop.get("result", ""))

        return True

    def _handle_push(self, args: dict) -> bool:
        if self.current.depth >= self.MAX_DEPTH:
            self.current.messages.append({
                "role": "user",
                "content": f"Cannot push: max depth ({self.MAX_DEPTH}) reached.",
            })
            return True
        self.stack.append(self.current)
        slug = args.get("name", "unnamed")
        objective = args.get("objective", "")
        context = args.get("context", "")
        return_spec = args.get("return_spec", "your findings")
        child_regs = _init_child_registers(
            objective=objective, return_spec=return_spec, schema=self.register_schema,
        )
        child = Frame(
            name=slug, objective=objective, context=context,
            return_spec=return_spec, depth=self.current.depth + 1,
            registers=child_regs,
        )
        child.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"## Objective\n{objective}\n\n"
                f"## Context\n{context}\n\n"
                f"## What to Return\n"
                f"When done, call pop_frame(result=...) with: {return_spec}"
            )},
        ]
        self.current = child
        self.consecutive_stalls = 0
        return True

    def _handle_pop(self, result: str) -> bool:
        if not self.stack:
            print(f"\n{'='*60}")
            print(f"ROOT POP -- Final result:\n{result[:500]}")
            self.final_result = result
            self.done = True
            return False
        completed_obj = self.current.objective
        parent = self.stack.pop()
        parent.messages.append({
            "role": "user",
            "content": (
                f"## Sub-task completed: '{completed_obj}'\n\n"
                f"### Result\n{result}"
            ),
        })
        self.current = parent
        self.consecutive_stalls = 0
        return True

    def run(self) -> str | None:
        while self.step():
            pass
        for probe in self.probes:
            if probe.trigger == "done":
                result = self._run_probe(probe.question)
                result["trigger"] = "done"
        return self.final_result


# ── Strategy wrapper ─────────────────────────────────────────────────


@register_strategy("stack_heap")
class StackHeapStrategy:
    """Stack+heap: agent-controlled context via push/pop + heap memory."""

    name = "stack_heap"

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 50,
        layout: dict | None = None,
        no_think_prefill: bool = False,
        probes: list | None = None,
        **kwargs,
    ) -> dict:
        agent = StackAgent(
            client, model, workspace="/testbed",
            tool_executor=executor,
            layout=layout,
            no_think_prefill=no_think_prefill,
        )
        agent.MAX_TOTAL_STEPS = max_steps
        if probes:
            agent.probes = probes
        agent.start(task)
        # Override return_spec for SWE-bench
        agent.current.return_spec = "The git diff of your fix and confirmation it doesn't break existing tests"
        agent.current.registers["R1"] = "The git diff of your fix and confirmation it doesn't break existing tests"

        t0 = time.monotonic()
        result = agent.run()
        elapsed = time.monotonic() - t0

        # Convert wall_clock to relative
        if agent.trace:
            t0_wall = agent.trace[0]["wall_clock"]
            for entry in agent.trace:
                entry["wall_clock_s"] = round(entry["wall_clock"] - t0_wall, 2)
                del entry["wall_clock"]

        return {
            "trace": agent.trace,
            "total_in": agent.total_input_tokens,
            "total_out": agent.total_output_tokens,
            "total_cached": agent.total_cached_tokens,
            "total_managed": agent.total_managed_tokens,
            "instruction_tokens": agent.instruction_tokens or 0,
            "elapsed_s": round(elapsed, 1),
            "max_depth": max((e["depth"] for e in agent.trace), default=0),
            "result": result,
            "peak_reg_bytes": agent.peak_reg_bytes,
            "probe_log": agent.probe_log,
            "heap_stats": {
                "peak_chars": max((e.get("heap_total_chars", 0) for e in agent.trace), default=0),
                "live_chunks": len(agent.heap),
                "freed_chunks": len(agent.heap_freed),
            },
            "layout": agent.layout.get("name", "unknown"),
            "masked_outputs": agent.total_masked_outputs,
        }
