"""synix-bench sweep — parallel multi-config runs."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Sweep config JSON")
@click.option("--out", "output_dir", default="results", help="Output directory")
@click.option("--workers", default=4, type=int, help="Parallel workers")
@click.option("-v", "--verbose", count=True)
def sweep(config_path: str, output_dir: str, workers: int, verbose: int) -> None:
    """Run multiple configs in parallel (sweep/matrix)."""
    import concurrent.futures
    import logging

    from synix.cli.engine import RunEngine
    from synix.core.config import RunConfig

    if verbose:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    sweep_config = json.loads(Path(config_path).read_text())

    # Sweep config format: {"configs": [RunConfig dict, ...]}
    configs = [RunConfig.from_dict(c) for c in sweep_config["configs"]]
    for c in configs:
        c.output_dir = output_dir
        c.llm = c.llm.resolve_env()

    console.print(f"Sweep: {len(configs)} configs, {workers} workers")

    def run_one(config: RunConfig) -> dict:
        engine = RunEngine(config)
        result = engine.run()
        return {
            "suite": result.suite,
            "strategy": result.strategy,
            "success_rate": result.success_rate,
            "tasks": len(result.tasks),
        }

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, c): i for i, c in enumerate(configs)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                console.print(f"  [{idx+1}/{len(configs)}] {result['strategy']}: {result['success_rate']:.1%}")
            except Exception as e:
                console.print(f"  [{idx+1}/{len(configs)}] ERROR: {e}", style="red")

    console.print(f"\n[bold green]Sweep complete![/bold green] {len(results)}/{len(configs)} succeeded")
