"""synix-bench run — execute a single benchmark run."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--suite", required=True, type=click.Choice(["swebench", "lens"]), help="Benchmark suite")
@click.option("--strategy", "--adapter", "strategy", default=None, help="Strategy (swebench) or adapter (lens) name")
@click.option("--dataset", default=None, type=click.Path(exists=True), help="Dataset path")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config JSON file")
@click.option("--out", "output_dir", default="results", help="Output directory")
@click.option("--model", default=None, help="LLM model name")
@click.option("--base-url", default=None, help="LLM API base URL")
@click.option("--api-key-env", default=None, help="Environment variable name for API key")
@click.option("--seed", default=42, type=int)
@click.option("--instance-id", default=None, help="Specific SWE-bench instance ID")
@click.option("--sample", default=None, type=int, help="Number of instances to sample")
@click.option("--trials", default=1, type=int, help="Number of trials per config")
@click.option("--max-steps", default=30, type=int, help="Max agent steps (swebench)")
@click.option("--timeout", default=1800, type=int, help="Timeout per task in seconds")
@click.option("--workers", default=6, type=int, help="Parallel workers")
@click.option("--budget", default="standard", type=click.Choice(["fast", "standard", "extended", "constrained-4k", "constrained-2k"]))
@click.option("--no-think-prefill", is_flag=True, help="Suppress Qwen3 think mode")
@click.option("--layout", default=None, type=click.Path(exists=True), help="Layout config (for stack+heap)")
@click.option("--provider", default=None, help="LLM provider (mock, openai)")
@click.option("--infra", "infra_provider", default="local", type=click.Choice(["local", "modal"]), help="Infrastructure provider")
@click.option("--skip-deploy", is_flag=True, help="Skip Modal deploy, just health-check existing endpoint")
@click.option("--parallel-questions", default=None, type=int, help="Concurrent questions (lens)")
@click.option("--cache-dir", default=None, type=click.Path(), help="Adapter cache dir (lens)")
@click.option("--prebuild-only", is_flag=True, help="Build SWE-bench container images and exit")
@click.option("-v", "--verbose", count=True)
def run(
    suite: str,
    strategy: str | None,
    dataset: str | None,
    config_path: str | None,
    output_dir: str,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    seed: int,
    instance_id: str | None,
    sample: int | None,
    trials: int,
    max_steps: int,
    timeout: int,
    workers: int,
    budget: str,
    no_think_prefill: bool,
    layout: str | None,
    provider: str | None,
    infra_provider: str,
    skip_deploy: bool,
    prebuild_only: bool,
    parallel_questions: int | None,
    cache_dir: str | None,
    verbose: int,
) -> None:
    """Run a benchmark suite against a strategy/adapter."""
    import logging
    import os

    from synix.cli.engine import RunEngine
    from synix.core.config import (
        AgentBudgetConfig,
        InfraConfig,
        LLMConfig,
        RunConfig,
        SWEBenchConfig,
    )

    if verbose:
        logging.basicConfig(level=logging.DEBUG if verbose > 1 else logging.INFO)

    # Build config from file or CLI args
    if config_path:
        config_data = json.loads(Path(config_path).read_text())
        config = RunConfig.from_dict(config_data)
    else:
        # Resolve API key
        api_key = None
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise click.ClickException(f"Environment variable {api_key_env} not set")

        extra_body = None
        if no_think_prefill:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        config = RunConfig(
            suite=suite,
            strategy=strategy or ("null" if suite == "lens" else "naive"),
            dataset=dataset or "",
            output_dir=output_dir,
            llm=LLMConfig(
                provider=provider or "openai",
                model=model or "gpt-4o-mini",
                api_base=base_url,
                api_key=api_key,
                seed=seed,
                extra_body=extra_body,
            ),
            infra=InfraConfig(),
            seed=seed,
            agent_budget=AgentBudgetConfig.from_preset(budget),
            swebench=SWEBenchConfig(
                max_steps=max_steps,
                timeout=timeout,
                workers=workers,
                layout_file=layout,
                no_think_prefill=no_think_prefill,
                prebuild_only=prebuild_only,
            ),
            sample=sample,
            trials=trials,
            instance_id=instance_id,
        )

    # CLI overrides
    if parallel_questions is not None:
        config.parallel_questions = parallel_questions
    if cache_dir is not None:
        config.cache_dir = cache_dir

    # Modal infrastructure
    config.infra.provider = infra_provider
    if skip_deploy:
        config.infra.modal_skip_deploy = True

    config.llm = config.llm.resolve_env()

    modal_prov = None
    if config.infra.provider == "modal":
        from synix.infra import create_provider

        modal_prov = create_provider(config.infra)
        if modal_prov is None:
            raise click.ClickException("Failed to create Modal provider")

        if not config.infra.modal_skip_deploy:
            console.print("[bold]Deploying Modal inference...[/bold]")
            endpoint = modal_prov.deploy(timeout=config.infra.modal_timeout)
        else:
            endpoint = modal_prov.get_endpoint()
            console.print(f"[bold]Using existing Modal endpoint:[/bold] {endpoint}")

        # Override LLM config to point at Modal endpoint
        config.llm = LLMConfig(
            provider=config.llm.provider if config.llm.provider != "mock" else "openai",
            model=config.llm.model,
            api_base=endpoint,
            api_key=modal_prov.api_token or config.llm.api_key,
            seed=config.llm.seed,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            extra_body=config.llm.extra_body,
        )

    try:
        engine = RunEngine(config)
        result = engine.run()
    finally:
        if modal_prov and not config.infra.modal_skip_deploy:
            console.print("[dim]Tearing down Modal apps...[/dim]")
            modal_prov.teardown()

    console.print(f"\n[bold green]Run complete![/bold green]")
    console.print(f"Suite: {result.suite}, Strategy: {result.strategy}")
    console.print(f"Tasks: {len(result.tasks)}, Success rate: {result.success_rate:.1%}")
    console.print(f"Output: {output_dir}")
