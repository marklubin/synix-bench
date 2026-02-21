"""Local process tool executor for synix-bench.

Ported from hybrid-memory-bench/src/hybrid_memory/tools.py with imports
adapted to the synix namespace. Executes agent tools in a sandboxed
local workspace directory.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from synix.core.errors import ExecutorError

log = logging.getLogger(__name__)

# Commands allowed via run_command
ALLOWED_COMMANDS = {"pytest", "python", "ls", "pip", "cat", "head", "tail", "find", "grep"}

MAX_OUTPUT_CHARS = 10_000

# OpenAI function-calling schemas for tools provided by this executor
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from workspace root",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from workspace root",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full file content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the directory (default: workspace root)",
                        "default": ".",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the workspace. "
                "Only allowed commands: pytest, python, ls, pip, cat, head, tail, find, grep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a text pattern in files under the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative directory to search in (default: workspace root)",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


def _resolve_path(workspace: Path, relative: str) -> Path:
    """Resolve a relative path against the workspace, with sandbox check."""
    resolved = (workspace / relative).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise PermissionError(f"Path escapes workspace: {relative}")
    return resolved


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate output to limit, with a note if truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


class LocalExecutor:
    """Execute agent tools in a sandboxed local workspace directory.

    Implements the ToolExecutor protocol defined in synix.executor.base.
    """

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).resolve()

    def __call__(self, name: str, args: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if name == "read_file":
                return self._tool_read_file(args)
            elif name == "write_file":
                return self._tool_write_file(args)
            elif name == "list_dir":
                return self._tool_list_dir(args)
            elif name == "run_command":
                return self._tool_run_command(args)
            elif name == "search_files":
                return self._tool_search_files(args)
            else:
                return f"Error: unknown tool '{name}'"
        except Exception as e:
            log.warning("Tool %s failed: %s", name, e)
            return f"Error: {e}"

    def get_patch(self) -> str:
        """Return git diff from the workspace. Returns empty string if not a git repo or no changes."""
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception as e:
            log.warning("get_patch failed: %s", e)
            return ""

    def _tool_read_file(self, args: dict) -> str:
        path = _resolve_path(self.workspace, args["path"])
        if not path.exists():
            return f"Error: file not found: {args['path']}"
        if not path.is_file():
            return f"Error: not a file: {args['path']}"
        content = path.read_text(errors="replace")
        return _truncate(content)

    def _tool_write_file(self, args: dict) -> str:
        path = _resolve_path(self.workspace, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return f"Written {len(args['content'])} chars to {args['path']}"

    def _tool_list_dir(self, args: dict) -> str:
        rel = args.get("path", ".")
        path = _resolve_path(self.workspace, rel)
        if not path.exists():
            return f"Error: directory not found: {rel}"
        if not path.is_dir():
            return f"Error: not a directory: {rel}"
        entries = sorted(path.iterdir())
        lines = []
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        if not lines:
            return "(empty directory)"
        return "\n".join(lines)

    def _tool_run_command(self, args: dict) -> str:
        command = args["command"].strip()

        # Strip "cd <path> && " prefixes -- workspace is already the cwd
        command = re.sub(r'^cd\s+\S+\s*&&\s*', '', command).strip()

        # Extract the base command (first word) for allowlist check
        base_cmd = command.split()[0] if command else ""
        # Strip path prefixes for allowlist check
        base_cmd_name = os.path.basename(base_cmd)

        if base_cmd_name not in ALLOWED_COMMANDS:
            return (
                f"Error: command '{base_cmd_name}' not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        # Route pytest, python, and pip through the current interpreter so
        # uv-managed packages are available in the workspace subprocess.
        if base_cmd_name == "pytest":
            command = command.replace(base_cmd, f"{sys.executable} -m pytest", 1)
        elif base_cmd_name == "python":
            command = command.replace(base_cmd, sys.executable, 1)
        elif base_cmd_name == "pip":
            command = command.replace(base_cmd, f"{sys.executable} -m pip", 1)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=120,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(self.workspace / "src"),
                },
            )
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            output_parts.append(f"(exit code: {result.returncode})")
            return _truncate("\n".join(output_parts))
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120 seconds"

    def _tool_search_files(self, args: dict) -> str:
        pattern = args["pattern"]
        rel = args.get("path", ".")
        search_dir = _resolve_path(self.workspace, rel)
        if not search_dir.exists():
            return f"Error: directory not found: {rel}"

        # Use grep -rn for search
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.txt", "--include=*.md",
                 "--include=*.json", "--include=*.yaml", "--include=*.yml",
                 "--include=*.toml", "--include=*.cfg", "--include=*.ini",
                 pattern, str(search_dir)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace),
            )
            output = result.stdout
            if not output:
                return f"No matches found for '{pattern}'"
            # Make paths relative to workspace
            workspace_str = str(self.workspace) + "/"
            output = output.replace(workspace_str, "")
            return _truncate(output)
        except subprocess.TimeoutExpired:
            return "Error: search timed out after 30 seconds"
