"""LENS memory adapters.

Re-exports the core adapter interface and registry for convenience.
"""
from synix.suites.lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    FilterField,
    MemoryAdapter,
    SearchResult,
)
from synix.suites.lens.adapters.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
)

__all__ = [
    "CapabilityManifest",
    "Document",
    "ExtraTool",
    "FilterField",
    "MemoryAdapter",
    "SearchResult",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
