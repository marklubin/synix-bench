"""synix-bench list — list available suites, strategies, adapters, metrics."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command("list")
@click.argument("what", type=click.Choice(["suites", "strategies", "adapters", "metrics"]))
def list_cmd(what: str) -> None:
    """List available suites, strategies, adapters, or metrics."""
    if what == "suites":
        from synix.suites.base import list_suites
        suites = list_suites()
        table = Table(title="Registered Suites")
        table.add_column("Name", style="bold cyan")
        for name in suites:
            table.add_row(name)
        console.print(table)

    elif what == "strategies":
        from synix.suites.swebench.strategies.base import list_strategies
        strategies = list_strategies()
        table = Table(title="SWE-bench Strategies")
        table.add_column("Name", style="bold cyan")
        for name in strategies:
            table.add_row(name)
        console.print(table)

    elif what == "adapters":
        try:
            from synix.suites.lens.adapters.registry import list_adapters
            adapter_map = list_adapters()
            table = Table(title="LENS Adapters")
            table.add_column("Name", style="bold cyan")
            table.add_column("Class", style="dim")
            for name, cls in sorted(adapter_map.items()):
                table.add_row(name, cls.__name__)
            console.print(table)
        except ImportError:
            console.print("[yellow]LENS adapters not available[/yellow]")

    elif what == "metrics":
        try:
            from synix.scorer.registry import list_metrics
            metric_map = list_metrics()
            table = Table(title="Scoring Metrics")
            table.add_column("Name", style="bold cyan")
            table.add_column("Tier", style="bold")
            for name, cls in sorted(metric_map.items()):
                instance = cls()
                table.add_row(name, str(instance.tier))
            console.print(table)
        except ImportError:
            console.print("[yellow]Scorer not available[/yellow]")
