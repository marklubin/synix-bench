"""synix-bench compare — compare results across runs."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.argument("results_files", nargs=-1, type=click.Path(exists=True))
@click.option("--out", "output_path", default=None, help="Output comparison file")
@click.option("-v", "--verbose", count=True)
def compare(results_files: tuple[str, ...], output_path: str | None, verbose: int) -> None:
    """Compare multiple benchmark results."""
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    if len(results_files) < 2:
        raise click.ClickException("Need at least 2 result files to compare")

    from synix.core.models import SuiteResult

    results = []
    for path in results_files:
        data = json.loads(Path(path).read_text())
        results.append(SuiteResult.from_dict(data))

    table = Table(title="Run Comparison")
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Tasks", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Total Tokens", justify="right")

    for r in results:
        table.add_row(
            r.strategy,
            r.suite,
            r.model,
            str(len(r.tasks)),
            f"{r.success_rate:.1%}",
            f"{r.total_tokens:,}",
        )

    console.print(table)

    if output_path:
        comparison = {
            "runs": [
                {
                    "strategy": r.strategy,
                    "suite": r.suite,
                    "model": r.model,
                    "tasks": len(r.tasks),
                    "success_rate": r.success_rate,
                    "total_tokens": r.total_tokens,
                }
                for r in results
            ]
        }
        Path(output_path).write_text(json.dumps(comparison, indent=2))
        console.print(f"\nComparison saved to {output_path}")
