"""Integration test: SWE-bench strategies with mock executor + mock LLM.

Mirrors hybrid-memory-bench/scripts/test_new_strategies.py to verify
the ported strategies produce valid traces without needing podman or API keys.
"""
from __future__ import annotations

import json

import pytest

from synix.suites.swebench.strategies.base import get_strategy, list_strategies


class MockExecutor:
    """Simulates a container executor with canned responses."""

    def __init__(self):
        self.call_count = 0
        self._patch = ""
        self.files = {
            "src/utils.py": (
                "def add(a, b):\n    return a + b\n\n"
                "def subtract(a, b):\n    return a - b\n"
            ),
            "tests/test_utils.py": (
                "from src.utils import add, subtract\n\n"
                "def test_add():\n    assert add(1, 2) == 3\n\n"
                "def test_subtract():\n    assert subtract(5, 3) == 2\n"
            ),
        }

    def __call__(self, tool_name: str, args: dict) -> str:
        self.call_count += 1
        if tool_name == "read_file":
            path = args.get("path", "")
            if path in self.files:
                return self.files[path]
            return f"Error: file not found: {path}"
        elif tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            self.files[path] = content
            self._patch += f"diff --git a/{path} b/{path}\n+{content[:100]}\n"
            return f"Wrote {len(content)} bytes to {path}"
        elif tool_name == "run_command":
            cmd = args.get("command", "")
            if "pytest" in cmd:
                return "===== 2 passed in 0.01s ====="
            elif "ls" in cmd:
                return "\n".join(self.files.keys())
            elif "git diff" in cmd:
                return self._patch or "(no changes)"
            return f"$ {cmd}\n(mock output)"
        elif tool_name == "list_files":
            path = args.get("path", ".")
            return "\n".join(
                f for f in self.files
                if f.startswith(path.rstrip("/")) or path == "."
            )
        return f"Unknown tool: {tool_name}"

    def get_patch(self) -> str:
        return self._patch


class MockOpenAIClient:
    """Minimal mock that returns tool calls then a finish response."""

    def __init__(self):
        self._call_count = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        """Return a mock response that makes one tool call then stops."""
        self._call_count += 1

        # After a few calls, stop making tool calls to end the loop
        if self._call_count > 3:
            return _MockResponse(content="Done. All tests pass.", tool_calls=None)

        # Make a read_file tool call
        return _MockResponse(
            content=None,
            tool_calls=[
                _MockToolCall(
                    id=f"call_{self._call_count}",
                    function=_MockFunction(
                        name="read_file",
                        arguments=json.dumps({"path": "src/utils.py"}),
                    ),
                )
            ],
        )


class _MockResponse:
    def __init__(self, content, tool_calls):
        self.choices = [_MockChoice(content, tool_calls)]
        self.usage = _MockUsage()


class _MockChoice:
    def __init__(self, content, tool_calls):
        self.message = _MockMessage(content, tool_calls)
        self.finish_reason = "stop" if tool_calls is None else "tool_calls"


class _MockMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"

    def model_dump(self):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return d


class _MockToolCall:
    def __init__(self, id, function):
        self.id = id
        self.type = "function"
        self.function = function


class _MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _MockPromptTokensDetails:
    cached_tokens = 0


class _MockUsage:
    prompt_tokens = 500
    completion_tokens = 100
    total_tokens = 600
    prompt_tokens_details = _MockPromptTokensDetails()


TASK = """Fix the bug in src/utils.py where the multiply function is missing.
The repository is at /testbed. Add a multiply(a, b) function and make sure tests pass.
Use read_file, write_file, run_command, and list_files tools."""


class TestSWEBenchStrategies:
    """Verify all 10 strategies load and the basic ones run with mock executor."""

    def test_all_strategies_registered(self):
        """All 10 strategies should be registered."""
        strategies = list_strategies()
        expected = {
            "naive", "window", "truncation", "summary", "masking",
            "rag", "incremental_summary", "structured_summary",
            "hierarchical", "stack_heap",
        }
        assert set(strategies) == expected

    @pytest.mark.parametrize("strategy_name", [
        "naive",
        "window",
        "truncation",
        "masking",
    ])
    def test_basic_strategies_run(self, strategy_name: str):
        """Basic strategies should run with mock executor + mock LLM."""
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()

        client = MockOpenAIClient()
        executor = MockExecutor()

        result = strategy.run(
            client=client,
            model="mock-model",
            task=TASK,
            executor=executor,
            max_steps=5,
        )

        # Verify result structure
        assert "steps" in result or "trace" in result, (
            f"{strategy_name}: result must have 'steps' or 'trace'"
        )
        assert "input_tokens" in result or "total_in" in result, (
            f"{strategy_name}: result must have token counts"
        )

        # Executor should have been called at least once
        assert executor.call_count > 0, f"{strategy_name}: no tool calls made"

    def test_strategy_classes_have_name(self):
        """Each strategy class should have a name attribute."""
        for name in list_strategies():
            strategy_cls = get_strategy(name)
            strategy = strategy_cls()
            assert hasattr(strategy, "name"), f"Strategy {name} missing 'name' attribute"
            assert strategy.name == name, (
                f"Strategy registered as {name!r} but has name={strategy.name!r}"
            )
