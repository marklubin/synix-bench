"""synix-bench report — generate reports from results."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--results", "results_path", required=True, type=click.Path(exists=True), help="Results JSON or directory")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json", "markdown"]))
@click.option("--out", "output_path", default=None, help="Output file path")
@click.option("-v", "--verbose", count=True)
def report(results_path: str, fmt: str, output_path: str | None, verbose: int) -> None:
    """Generate a report from benchmark results."""
    import json
    import logging
    from pathlib import Path

    if verbose:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    from synix.core.models import SuiteResult

    results_file = Path(results_path)
    if results_file.is_dir():
        json_files = sorted(results_file.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            raise click.ClickException(f"No JSON files found in {results_path}")
        results_file = json_files[0]

    data = json.loads(results_file.read_text())
    result = SuiteResult.from_dict(data)

    lines = [
        f"# synix-bench Report",
        f"",
        f"Suite: {result.suite}",
        f"Strategy: {result.strategy}",
        f"Model: {result.model}",
        f"Tasks: {len(result.tasks)}",
        f"Success rate: {result.success_rate:.1%}",
        f"Total tokens: {result.total_tokens:,}",
        f"",
        f"## Per-task results",
        f"",
    ]
    for task in result.tasks:
        status = "PASS" if task.success else "FAIL"
        tokens = task.total_input_tokens + task.total_output_tokens
        lines.append(f"- [{status}] {task.task_id} ({len(task.steps)} steps, {tokens:,} tokens)")

    report_text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report_text)
        console.print(f"Report saved to {output_path}")
    else:
        console.print(report_text)
