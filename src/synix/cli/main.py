"""synix-bench CLI entry point."""

from __future__ import annotations

import click
from rich.console import Console

from synix import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="synix-bench")
def cli() -> None:
    """synix-bench: Unified benchmark for agent memory and context management."""


# Import and register subcommands
from synix.cli.run import run  # noqa: E402
from synix.cli.sweep import sweep  # noqa: E402
from synix.cli.score import score  # noqa: E402
from synix.cli.report import report  # noqa: E402
from synix.cli.compare import compare  # noqa: E402
from synix.cli.list import list_cmd  # noqa: E402
from synix.cli.smoke import smoke  # noqa: E402

cli.add_command(run)
cli.add_command(sweep)
cli.add_command(score)
cli.add_command(report)
cli.add_command(compare)
cli.add_command(list_cmd, name="list")
cli.add_command(smoke)


if __name__ == "__main__":
    cli()
