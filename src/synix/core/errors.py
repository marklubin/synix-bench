"""Synix error hierarchy.

Combines LENS error types with SWE-bench specific errors.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Generator


class SynixError(Exception):
    """Base exception for all synix-bench errors."""


class ConfigError(SynixError):
    """Invalid or missing configuration."""


class AdapterError(SynixError):
    """Error in adapter execution."""


class StrategyError(SynixError):
    """Error in context strategy execution."""


class BudgetExceededError(SynixError):
    """Adapter or agent exceeded its LLM budget (calls or tokens)."""

    def __init__(self, phase: str, method: str, limit_kind: str, limit: int, actual: int) -> None:
        self.phase = phase
        self.method = method
        self.limit_kind = limit_kind
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"Budget exceeded in {phase}/{method}: "
            f"{limit_kind} limit={limit}, actual={actual}"
        )


class LatencyExceededError(SynixError):
    """Adapter exceeded latency cap for a method."""

    def __init__(self, method: str, limit_ms: float, actual_ms: float) -> None:
        self.method = method
        self.limit_ms = limit_ms
        self.actual_ms = actual_ms
        super().__init__(
            f"Latency exceeded in {method}: limit={limit_ms}ms, actual={actual_ms:.1f}ms"
        )


class ValidationError(SynixError):
    """Output schema or evidence validation failure."""


class EvidenceError(ValidationError):
    """Evidence quote not found as exact substring in episode."""

    def __init__(self, episode_id: str, quote: str) -> None:
        self.episode_id = episode_id
        self.quote = quote
        super().__init__(
            f"Evidence quote not found in episode {episode_id}: {quote[:80]!r}..."
        )


class AntiCheatError(SynixError):
    """Adapter attempted to access episode text at query time."""


class DatasetError(SynixError):
    """Invalid or missing dataset."""


class ScoringError(SynixError):
    """Error during scoring."""


class PluginError(SynixError):
    """Error loading adapter or metric plugin."""


class ContainerError(SynixError):
    """Error in container execution (podman/docker)."""


class ExecutorError(SynixError):
    """Error executing a tool."""


@contextlib.contextmanager
def atomic_write(path: Path | str) -> Generator[Path, None, None]:
    """Write to a temp file then atomically rename to target path.

    Usage:
        with atomic_write("output.json") as tmp:
            tmp.write_text(json.dumps(data))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_path)
    try:
        os.close(fd)
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
