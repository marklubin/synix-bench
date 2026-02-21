"""synix-bench sweep — parallel multi-config runs."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


def _run_one(config) -> dict:
    """Run a single config — must be module-level for ProcessPoolExecutor pickling."""
    from synix.cli.engine import RunEngine

    engine = RunEngine(config)
    result = engine.run()
    return {
        "suite": result.suite,
        "strategy": result.strategy,
        "success_rate": result.success_rate,
        "tasks": len(result.tasks),
    }


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Sweep config JSON")
@click.option("--out", "output_dir", default="results", help="Output directory")
@click.option("--workers", default=4, type=int, help="Parallel workers")
@click.option("--provider", "infra_provider", default="local", type=click.Choice(["local", "modal"]), help="Infrastructure provider")
@click.option("--skip-deploy", is_flag=True, help="Skip Modal deploy, use existing endpoint")
@click.option("--keep-alive", is_flag=True, help="Don't teardown Modal apps after sweep")
@click.option("-v", "--verbose", count=True)
def sweep(
    config_path: str,
    output_dir: str,
    workers: int,
    infra_provider: str,
    skip_deploy: bool,
    keep_alive: bool,
    verbose: int,
) -> None:
    """Run multiple configs in parallel (sweep/matrix)."""
    import concurrent.futures
    import logging

    from synix.cli.engine import RunEngine
    from synix.core.config import LLMConfig, RunConfig

    if verbose:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    sweep_config = json.loads(Path(config_path).read_text())

    # Sweep config format: {"configs": [RunConfig dict, ...]}
    configs = [RunConfig.from_dict(c) for c in sweep_config["configs"]]
    for c in configs:
        c.output_dir = output_dir
        c.llm = c.llm.resolve_env()

    # Modal infrastructure — deploy once, share endpoint across all configs
    modal_prov = None
    if infra_provider == "modal":
        from synix.core.config import InfraConfig
        from synix.infra import create_provider

        infra_cfg = InfraConfig(
            provider="modal",
            modal_skip_deploy=skip_deploy,
        )
        modal_prov = create_provider(infra_cfg)
        if modal_prov is None:
            raise click.ClickException("Failed to create Modal provider")

        if not skip_deploy:
            console.print("[bold]Deploying Modal inference...[/bold]")
            endpoint = modal_prov.deploy(timeout=infra_cfg.modal_timeout)
        else:
            endpoint = modal_prov.get_endpoint()
            console.print(f"[bold]Using existing Modal endpoint:[/bold] {endpoint}")

        # Override all configs to use Modal endpoint
        for c in configs:
            c.llm = LLMConfig(
                provider=c.llm.provider if c.llm.provider != "mock" else "openai",
                model=c.llm.model,
                api_base=endpoint,
                api_key=modal_prov.api_token or c.llm.api_key,
                seed=c.llm.seed,
                temperature=c.llm.temperature,
                max_tokens=c.llm.max_tokens,
                extra_body=c.llm.extra_body,
            )

    console.print(f"Sweep: {len(configs)} configs, {workers} workers")

    results = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, c): i for i, c in enumerate(configs)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    console.print(f"  [{idx+1}/{len(configs)}] {result['strategy']}: {result['success_rate']:.1%}")
                except Exception as e:
                    console.print(f"  [{idx+1}/{len(configs)}] ERROR: {e}", style="red")
    finally:
        if modal_prov and not keep_alive:
            console.print("[dim]Tearing down Modal apps...[/dim]")
            modal_prov.teardown()
        elif modal_prov and keep_alive:
            console.print(f"[dim]Keeping Modal apps alive (endpoint: {modal_prov.get_endpoint()})[/dim]")

    console.print(f"\n[bold green]Sweep complete![/bold green] {len(results)}/{len(configs)} succeeded")
