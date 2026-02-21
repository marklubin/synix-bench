"""SWE-bench agent tool schemas and dispatch.

NAIVE_TOOLS: the OpenAI function-calling tool schemas used by all
non-stack strategies. Stack+heap uses its own expanded schema.
"""

from __future__ import annotations

NAIVE_TOOLS = [
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
]

NAIVE_SYSTEM = (
    "You are a skilled software engineer. Complete the task using the "
    "provided tools. Be thorough and efficient."
)
