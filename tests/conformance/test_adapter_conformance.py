"""Conformance test skeleton for memory adapters.

Verifies that adapter implementations correctly implement the MemoryAdapter
protocol: reset, ingest, prepare, search, retrieve, get_capabilities.
"""
from __future__ import annotations

import pytest

from synix.suites.lens.adapters.base import CapabilityManifest, MemoryAdapter
from synix.suites.lens.adapters.null import NullAdapter
from synix.suites.lens.adapters.sqlite import SQLiteAdapter


CONFORMANCE_ADAPTERS = [
    NullAdapter,
    SQLiteAdapter,
]


@pytest.fixture(params=CONFORMANCE_ADAPTERS, ids=lambda c: c.__name__)
def adapter(request) -> MemoryAdapter:
    return request.param()


class TestAdapterConformance:
    """Verify adapters implement the MemoryAdapter interface correctly."""

    def test_reset_accepts_scope_id(self, adapter: MemoryAdapter):
        adapter.reset("test_scope")

    def test_ingest_after_reset(self, adapter: MemoryAdapter):
        adapter.reset("test_scope")
        adapter.ingest("ep_001", "test_scope", "2024-01-01T00:00:00", "Episode text")

    def test_prepare_after_ingest(self, adapter: MemoryAdapter):
        adapter.reset("test_scope")
        adapter.ingest("ep_001", "test_scope", "2024-01-01T00:00:00", "Episode text")
        adapter.prepare("test_scope", 1)

    def test_search_returns_list(self, adapter: MemoryAdapter):
        adapter.reset("test_scope")
        adapter.ingest("ep_001", "test_scope", "2024-01-01T00:00:00", "Episode text")
        adapter.prepare("test_scope", 1)
        results = adapter.search("episode")
        assert isinstance(results, list)

    def test_retrieve_returns_document_or_none(self, adapter: MemoryAdapter):
        adapter.reset("test_scope")
        result = adapter.retrieve("nonexistent_id")
        assert result is None

    def test_get_capabilities_returns_manifest(self, adapter: MemoryAdapter):
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert isinstance(caps.search_modes, list)
        assert isinstance(caps.max_results_per_search, int)
