"""ToolExecutor protocol.

Shared interface for executing agent tool calls across different
environments (containers, local process, cloud).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for executing agent tool calls."""

    def __call__(self, name: str, args: dict) -> str:
        """Execute a tool and return its output as a string."""
        ...

    def get_patch(self) -> str:
        """Return the current git diff / patch from the execution environment."""
        ...
