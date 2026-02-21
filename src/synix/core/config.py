"""Unified configuration for synix-bench.

Merges LENS RunConfig/LLMConfig/AgentBudgetConfig with HMB's SWE-bench
config knobs into a single hierarchy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the LLM used by the runner."""

    provider: str = "mock"  # "mock", "openai"
    model: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key: str | None = None
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int = 4096
    extra_body: dict | None = None  # e.g. {"chat_template_kwargs": {"enable_thinking": False}}

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def resolve_env(self) -> LLMConfig:
        """Override fields from environment variables."""
        return LLMConfig(
            provider=os.environ.get("SYNIX_LLM_PROVIDER", self.provider),
            model=os.environ.get("SYNIX_LLM_MODEL", self.model),
            api_base=os.environ.get("SYNIX_LLM_API_BASE", self.api_base),
            api_key=os.environ.get("SYNIX_LLM_API_KEY", self.api_key),
            seed=int(os.environ.get("SYNIX_LLM_SEED", str(self.seed))),
            temperature=float(os.environ.get("SYNIX_LLM_TEMPERATURE", str(self.temperature))),
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "provider": self.provider,
            "model": self.model,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            d["api_base"] = self.api_base
        if self.extra_body:
            d["extra_body"] = self.extra_body
        return d


@dataclass
class AgentBudgetConfig:
    """Budget configuration for the agent's per-question execution (LENS suite)."""

    preset: str = "standard"
    max_turns: int = 10
    max_tool_calls: int = 20
    max_payload_bytes: int = 65536
    max_latency_per_call_ms: float = 5000
    max_agent_tokens: int = 32768
    ingest_max_latency_ms: float = 200
    max_cumulative_result_tokens: int = 0  # 0 = unlimited

    @classmethod
    def fast(cls) -> AgentBudgetConfig:
        return cls(
            preset="fast", max_turns=5, max_tool_calls=10,
            max_payload_bytes=32768, max_agent_tokens=4096,
        )

    @classmethod
    def standard(cls) -> AgentBudgetConfig:
        return cls(preset="standard")

    @classmethod
    def extended(cls) -> AgentBudgetConfig:
        return cls(
            preset="extended", max_turns=20, max_tool_calls=50,
            max_payload_bytes=131072, max_latency_per_call_ms=10000,
            max_agent_tokens=65536,
        )

    @classmethod
    def constrained_4k(cls) -> AgentBudgetConfig:
        return cls(
            preset="constrained-4k", max_turns=6, max_tool_calls=12,
            max_agent_tokens=16384, max_cumulative_result_tokens=4096,
        )

    @classmethod
    def constrained_2k(cls) -> AgentBudgetConfig:
        return cls(
            preset="constrained-2k", max_turns=6, max_tool_calls=12,
            max_agent_tokens=16384, max_cumulative_result_tokens=2048,
        )

    @classmethod
    def from_preset(cls, name: str) -> AgentBudgetConfig:
        presets = {
            "fast": cls.fast,
            "standard": cls.standard,
            "extended": cls.extended,
            "constrained-4k": cls.constrained_4k,
            "constrained-2k": cls.constrained_2k,
        }
        if name not in presets:
            msg = f"Unknown agent budget preset: {name!r}. Choose from: {list(presets)}"
            raise ValueError(msg)
        return presets[name]()

    @classmethod
    def from_dict(cls, d: dict) -> AgentBudgetConfig:
        preset = d.get("preset", "standard")
        config = cls.from_preset(preset)
        for key in cls.__dataclass_fields__:
            if key in d and key != "preset":
                setattr(config, key, d[key])
        return config


@dataclass
class SWEBenchConfig:
    """SWE-bench suite specific configuration."""

    max_steps: int = 30
    timeout: int = 1800
    workers: int = 6
    layout_file: str | None = None  # path to layout JSON (for stack+heap)
    no_think_prefill: bool = False  # Qwen3 think mode suppression
    prebuild_only: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> SWEBenchConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InfraConfig:
    """Infrastructure provider configuration."""

    provider: str = "local"  # "local", "modal"
    container_cmd: str = "podman"  # "podman" or "docker"
    docker_host: str | None = None  # e.g. "unix:///run/user/1000/podman/podman.sock"
    ghcr_prefix: str = "ghcr.io/marklubin/swebench"

    @classmethod
    def from_dict(cls, d: dict) -> InfraConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunConfig:
    """Top-level unified run configuration.

    Covers both suites. Suite-specific fields are in sub-configs
    (swebench_config, agent_budget for LENS).
    """

    suite: str = "lens"  # "swebench" or "lens"
    strategy: str = "null"  # strategy (swebench) or adapter (lens)
    dataset: str = ""
    output_dir: str = "results"
    llm: LLMConfig = field(default_factory=LLMConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    seed: int = 42

    # LENS-specific
    agent_budget: AgentBudgetConfig = field(default_factory=AgentBudgetConfig.standard)
    checkpoints: list[int] = field(default_factory=lambda: [10, 20, 40, 80])
    parallel_questions: int = 1
    cache_dir: str | None = None

    # SWE-bench-specific
    swebench: SWEBenchConfig = field(default_factory=SWEBenchConfig)

    # Multi-instance / sweep
    sample: int | None = None  # number of instances to sample
    trials: int = 1
    instance_id: str | None = None  # specific instance

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        llm = LLMConfig.from_dict(d["llm"]) if "llm" in d else LLMConfig()
        agent_budget = (
            AgentBudgetConfig.from_dict(d["agent_budget"])
            if "agent_budget" in d
            else AgentBudgetConfig.standard()
        )
        infra = InfraConfig.from_dict(d["infra"]) if "infra" in d else InfraConfig()
        swebench = SWEBenchConfig.from_dict(d["swebench"]) if "swebench" in d else SWEBenchConfig()

        return cls(
            suite=d.get("suite", "lens"),
            strategy=d.get("strategy", d.get("adapter", "null")),
            dataset=d.get("dataset", ""),
            output_dir=d.get("output_dir", "results"),
            llm=llm.resolve_env(),
            infra=infra,
            seed=d.get("seed", 42),
            agent_budget=agent_budget,
            checkpoints=d.get("checkpoints", [10, 20, 40, 80]),
            parallel_questions=d.get("parallel_questions", 1),
            cache_dir=d.get("cache_dir"),
            swebench=swebench,
            sample=d.get("sample"),
            trials=d.get("trials", 1),
            instance_id=d.get("instance_id"),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "suite": self.suite,
            "strategy": self.strategy,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "llm": self.llm.to_dict(),
            "seed": self.seed,
        }
        if self.suite == "lens":
            d["agent_budget"] = {
                "preset": self.agent_budget.preset,
                "max_turns": self.agent_budget.max_turns,
                "max_tool_calls": self.agent_budget.max_tool_calls,
                "max_agent_tokens": self.agent_budget.max_agent_tokens,
            }
            d["checkpoints"] = self.checkpoints
            if self.parallel_questions > 1:
                d["parallel_questions"] = self.parallel_questions
            if self.cache_dir:
                d["cache_dir"] = self.cache_dir
        if self.suite == "swebench":
            d["swebench"] = {
                "max_steps": self.swebench.max_steps,
                "timeout": self.swebench.timeout,
                "workers": self.swebench.workers,
            }
            if self.swebench.layout_file:
                d["swebench"]["layout_file"] = self.swebench.layout_file
        if self.sample:
            d["sample"] = self.sample
        if self.trials > 1:
            d["trials"] = self.trials
        if self.instance_id:
            d["instance_id"] = self.instance_id
        return d
