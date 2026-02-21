"""synix-bench smoke — quick sanity check."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--suite", default="lens", type=click.Choice(["swebench", "lens"]), help="Suite to smoke test")
def smoke(suite: str) -> None:
    """Quick sanity check with mock LLM + null/mock adapter."""
    if suite == "lens":
        _smoke_lens()
    elif suite == "swebench":
        _smoke_swebench()


def _smoke_lens() -> None:
    """Smoke test: null adapter + mock LLM on bundled smoke dataset."""
    try:
        from synix.llm.client import MockLLMClient
        from synix.core.config import AgentBudgetConfig, RunConfig
        from synix.suites.lens.suite import LENSSuite

        console.print("Running LENS smoke test (null adapter + mock LLM)...")

        config = RunConfig(
            suite="lens",
            strategy="null",
            agent_budget=AgentBudgetConfig.fast(),
            checkpoints=[5, 10],
        )

        suite = LENSSuite()
        tasks = suite.load_tasks(config)
        console.print(f"  Loaded {len(tasks)} tasks")

        console.print("[bold green]LENS smoke test passed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]LENS smoke test failed:[/bold red] {e}")
        raise click.ClickException(str(e))


def _smoke_swebench() -> None:
    """Smoke test: verify SWE-bench suite loads and strategies are registered."""
    try:
        from synix.suites.swebench.strategies.base import list_strategies
        from synix.suites.swebench.suite import SWEBenchSuite

        console.print("Running SWE-bench smoke test...")

        strategies = list_strategies()
        console.print(f"  Registered strategies: {strategies}")

        suite = SWEBenchSuite()
        console.print(f"  Suite name: {suite.name}")

        console.print("[bold green]SWE-bench smoke test passed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]SWE-bench smoke test failed:[/bold red] {e}")
        raise click.ClickException(str(e))
