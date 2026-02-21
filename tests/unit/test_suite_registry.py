"""Test that both suites are discoverable."""
from __future__ import annotations

from synix.suites.base import get_suite, list_suites


def test_lens_suite_registered():
    cls = get_suite("lens")
    assert cls.name == "lens"


def test_list_suites_includes_lens():
    suites = list_suites()
    assert "lens" in suites
