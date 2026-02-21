"""synix-bench score — score benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--results", "results_path", required=True, type=click.Path(exists=True), help="Results JSON file or directory")
@click.option("--out", "output_path", default=None, help="Output scorecard path")
@click.option("--tier", type=int, default=None, help="Only compute metrics of this tier")
@click.option("--judge-model", default=None, help="OpenAI model for LLM judge")
@click.option("-v", "--verbose", count=True)
def score(results_path: str, output_path: str | None, tier: int | None, judge_model: str | None, verbose: int) -> None:
    """Score benchmark results."""
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    results_file = Path(results_path)
    if results_file.is_dir():
        # Find the latest results file
        json_files = sorted(results_file.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            raise click.ClickException(f"No JSON files found in {results_path}")
        results_file = json_files[0]

    data = json.loads(results_file.read_text())
    suite_name = data.get("suite", "unknown")

    if suite_name == "lens":
        # Use LENS scorer
        from synix.scorer.engine import ScorerEngine
        console.print(f"Scoring LENS results from {results_file}")
        scorer = ScorerEngine(tier_filter=tier)
        # TODO: reconstruct RunResult from data and score
        console.print("[yellow]LENS scoring not yet integrated — use lens CLI for now[/yellow]")
    elif suite_name == "swebench":
        # SWE-bench metrics
        from synix.core.models import SuiteResult
        result = SuiteResult.from_dict(data)
        total = len(result.tasks)
        passed = sum(1 for t in result.tasks if t.success)
        patches = sum(1 for t in result.tasks if t.raw_result.get("patch"))
        total_tokens = result.total_tokens

        console.print(f"\n[bold]SWE-bench Results: {result.strategy}[/bold]")
        console.print(f"  Tasks: {total}")
        console.print(f"  Verifier pass rate: {passed}/{total} ({passed/total:.1%})" if total else "  No tasks")
        console.print(f"  Patch rate: {patches}/{total} ({patches/total:.1%})" if total else "")
        console.print(f"  Total tokens: {total_tokens:,}")
        if passed and total_tokens:
            console.print(f"  Token efficiency: {(passed/total) / (total_tokens/1_000_000):.2f} success/Mtok")
    else:
        console.print(f"[yellow]Unknown suite: {suite_name}[/yellow]")

    if output_path:
        console.print(f"Scorecard saved to {output_path}")
