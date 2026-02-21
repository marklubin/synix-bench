"""Test RunConfig.from_dict / to_dict roundtrip."""
from __future__ import annotations

from synix.core.config import RunConfig, LLMConfig, AgentBudgetConfig


class TestRunConfig:
    def test_defaults(self):
        config = RunConfig()
        assert config.suite == "lens"
        assert config.strategy == "null"
        assert config.llm.model is not None

    def test_roundtrip(self):
        config = RunConfig(
            suite="lens",
            strategy="null",
            llm=LLMConfig(model="gpt-4o", api_key="test-key"),
            agent_budget=AgentBudgetConfig(max_turns=5, max_tool_calls=10),
        )
        d = config.to_dict()
        restored = RunConfig.from_dict(d)
        assert restored.suite == "lens"
        assert restored.strategy == "null"
        assert restored.llm.model == "gpt-4o"
        assert restored.agent_budget.max_turns == 5
        assert restored.agent_budget.max_tool_calls == 10


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.model is not None
        assert config.temperature == 0.0

    def test_roundtrip(self):
        config = LLMConfig(model="claude-3-opus", temperature=0.5, max_tokens=4096)
        d = config.to_dict()
        restored = LLMConfig.from_dict(d)
        assert restored.model == "claude-3-opus"
        assert restored.temperature == 0.5
        assert restored.max_tokens == 4096
