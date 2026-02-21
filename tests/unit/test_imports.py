"""Verify all major modules are importable."""
from __future__ import annotations


def test_core_models():
    from synix.core.models import StepTrace, TaskResult, SuiteResult, VerificationResult


def test_core_config():
    from synix.core.config import RunConfig, LLMConfig, AgentBudgetConfig


def test_core_errors():
    from synix.core.errors import SynixError, AdapterError, AntiCheatError, PluginError


def test_suites_base():
    from synix.suites.base import BenchmarkSuite, get_suite, list_suites


def test_llm_client():
    from synix.llm.client import BaseLLMClient, MockLLMClient, ToolDefinition, ToolCall, ToolResult, AgentTurn


def test_executor_base():
    from synix.executor.base import ToolExecutor


def test_lens_suite():
    from synix.suites.lens.suite import LENSSuite


def test_lens_adapters():
    from synix.suites.lens.adapters import MemoryAdapter, SearchResult, Document, get_adapter, list_adapters


def test_lens_models():
    from synix.suites.lens.models import Episode, Question, GroundTruth, AgentAnswer, QuestionResult


def test_lens_anticheat():
    from synix.suites.lens.anticheat import EpisodeVault


def test_agent_harness():
    from synix.agent.harness import AgentHarness


def test_agent_tool_bridge():
    from synix.agent.tool_bridge import build_tool_definitions, dispatch_tool_call


def test_agent_budget():
    from synix.agent.budget import BudgetViolation, QuestionBudget, BudgetEnforcement


def test_metering():
    from synix.metering import MeteringUsage
    from synix.metering.manager import MeteringManager
    from synix.metering.proxy import UsageStore, create_proxy_server


def test_lens_suite_strategies():
    """LENSSuite.list_strategies() should return adapter names."""
    from synix.suites.lens.suite import LENSSuite
    suite = LENSSuite()
    strategies = suite.list_strategies()
    assert isinstance(strategies, list)
    assert "null" in strategies
    assert "sqlite" in strategies
    assert len(strategies) >= 10
