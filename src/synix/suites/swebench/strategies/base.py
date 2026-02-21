"""ContextStrategy protocol and registry for SWE-bench strategies.

Each of the 10 HMB run_*_loop() functions is wrapped as a class
implementing this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openai import OpenAI

from synix.executor.base import ToolExecutor

# Global strategy registry
_STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(name: str):
    """Decorator to register a ContextStrategy implementation."""
    def decorator(cls: type) -> type:
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> type:
    """Look up a registered strategy by name."""
    if name not in _STRATEGY_REGISTRY:
        _lazy_load_strategies()
    if name not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise KeyError(f"Unknown strategy: {name!r}. Available: {available}")
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    _lazy_load_strategies()
    return sorted(_STRATEGY_REGISTRY.keys())


def _lazy_load_strategies() -> None:
    """Import strategy modules to trigger registration."""
    import importlib
    for module_name in [
        "synix.suites.swebench.strategies.naive",
        "synix.suites.swebench.strategies.window",
        "synix.suites.swebench.strategies.truncation",
        "synix.suites.swebench.strategies.summary",
        "synix.suites.swebench.strategies.masking",
        "synix.suites.swebench.strategies.rag",
        "synix.suites.swebench.strategies.incremental",
        "synix.suites.swebench.strategies.structured",
        "synix.suites.swebench.strategies.hierarchical",
        "synix.suites.swebench.strategies.stack_heap",
    ]:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass


@runtime_checkable
class ContextStrategy(Protocol):
    """Protocol for SWE-bench context management strategies.

    Each strategy implements a run() method that executes the agent loop
    with its specific context management approach.
    """

    name: str

    def run(
        self,
        client: OpenAI,
        model: str,
        task: str,
        executor: ToolExecutor,
        max_steps: int = 30,
        **kwargs: Any,
    ) -> dict:
        """Run the agent loop and return a result dict.

        Args:
            client: OpenAI-compatible API client
            model: Model name/ID
            task: Task description / system prompt
            executor: Tool executor for running commands
            max_steps: Maximum agent steps
            **kwargs: Strategy-specific options (layout config, etc.)

        Returns:
            Dict with at minimum: steps, patch, input_tokens, output_tokens
        """
        ...
