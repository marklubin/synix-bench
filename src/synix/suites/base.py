"""BenchmarkSuite ABC and registry.

Each suite (swebench, lens) implements this interface. The RunEngine
dispatches to suites by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from synix.core.config import RunConfig
from synix.core.models import TaskResult, VerificationResult

# Global suite registry
_SUITE_REGISTRY: dict[str, type[BenchmarkSuite]] = {}


def register_suite(name: str):
    """Decorator to register a BenchmarkSuite implementation."""
    def decorator(cls: type[BenchmarkSuite]) -> type[BenchmarkSuite]:
        _SUITE_REGISTRY[name] = cls
        return cls
    return decorator


def get_suite(name: str) -> type[BenchmarkSuite]:
    """Look up a registered suite by name."""
    if name not in _SUITE_REGISTRY:
        # Try lazy imports
        _lazy_load_suites()
    if name not in _SUITE_REGISTRY:
        available = list(_SUITE_REGISTRY.keys())
        raise KeyError(f"Unknown suite: {name!r}. Available: {available}")
    return _SUITE_REGISTRY[name]


def list_suites() -> list[str]:
    """Return names of all registered suites."""
    _lazy_load_suites()
    return sorted(_SUITE_REGISTRY.keys())


def _lazy_load_suites() -> None:
    """Import suite modules to trigger registration."""
    import importlib
    for module_name in [
        "synix.suites.swebench.suite",
        "synix.suites.lens.suite",
    ]:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass


class BenchmarkSuite(ABC):
    """Abstract base class for benchmark suites."""

    name: str

    @abstractmethod
    def load_tasks(self, config: RunConfig) -> list[dict]:
        """Load tasks/instances for this suite.

        Returns a list of task dicts (suite-specific schema).
        """
        ...

    @abstractmethod
    def run_task(self, task: dict, config: RunConfig) -> TaskResult:
        """Run a single task and return the result."""
        ...

    @abstractmethod
    def verify(self, task: dict, result: TaskResult) -> VerificationResult:
        """Verify a task result (run eval scripts, check answers, etc.)."""
        ...

    def setup(self, config: RunConfig) -> None:
        """Optional setup (e.g., prebuild container images)."""

    def teardown(self) -> None:
        """Optional cleanup."""

    def list_strategies(self) -> list[str]:
        """Return available strategy/adapter names for this suite."""
        return []
