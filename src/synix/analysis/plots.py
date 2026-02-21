"""Visualization functions for strategy comparison.

Extracted from hybrid-memory-bench/scripts/analyze_matrix.py.
Provides forest_plot, per_repo, and token_comparison chart functions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Strategy display names and colors
STRATEGY_NAMES: dict[str, str] = {
    "naive": "Naive",
    "window": "Window (k=20)",
    "truncation": "Truncation (8K)",
    "summary": "Summary (int=5)",
    "masking": "Masking (w=10)",
    "order-conv-mask": "Stack+Heap (OCM)",
    "rag": "RAG (k=5, w=6)",
    "incremental_summary": "Incr. Summary",
    "structured_summary": "Struct. Summary",
    "hierarchical": "Hierarchical 3-Tier",
}

STRATEGY_COLORS: dict[str, str] = {
    "naive": "#F44336",
    "window": "#9C27B0",
    "truncation": "#00BCD4",
    "summary": "#CDDC39",
    "masking": "#FF9800",
    "order-conv-mask": "#FF5722",
    "rag": "#4CAF50",
    "incremental_summary": "#2196F3",
    "structured_summary": "#3F51B5",
    "hierarchical": "#795548",
}

STRATEGY_ORDER: list[str] = [
    "naive", "window", "truncation", "summary", "masking",
    "rag", "incremental_summary", "structured_summary", "hierarchical",
    "order-conv-mask",
]


def plot_forest(comparisons: list[dict], output_path: Path) -> None:
    """Forest plot showing paired differences with confidence intervals."""
    if not comparisons:
        logger.warning("No comparisons to plot for forest chart")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(comparisons) * 1.2)))

    y_positions = list(range(len(comparisons)))
    labels = []
    for i, comp in enumerate(comparisons):
        strategy = comp["strategy_b"]
        display_name = STRATEGY_NAMES.get(strategy, strategy)
        color = STRATEGY_COLORS.get(strategy, "#666666")

        diff = comp["observed_diff"]
        ci_low = comp["ci_low"]
        ci_high = comp["ci_high"]
        sig = comp["significant"]

        # Point estimate + CI
        ax.errorbar(
            diff, i,
            xerr=[[diff - ci_low], [ci_high - diff]],
            fmt="o" if not sig else "D",
            color=color,
            markersize=10 if sig else 8,
            capsize=6,
            capthick=2,
            linewidth=2,
            markeredgecolor="black" if sig else color,
            markeredgewidth=1.5 if sig else 0.5,
        )

        labels.append(display_name)

        ax.annotate(
            f"{diff:+.1%} p={comp['p_value']:.3f}",
            (ci_high + 0.02, i),
            fontsize=9,
            va="center",
        )

    # Zero line
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    control_name = STRATEGY_NAMES.get(comparisons[0]["strategy_a"], comparisons[0]["strategy_a"])
    ax.set_xlabel(f"Difference in Patch Production Rate vs {control_name}", fontsize=12)
    ax.set_title("Paired Bootstrap: Strategy Differences\n(Bonferroni-corrected 95% CI)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Forest plot saved: %s", output_path)


def plot_per_repo(
    repo_stats: dict[str, dict[str, dict]],
    strategies: list[str],
    output_path: Path,
) -> None:
    """Grouped bar chart of per-repo solve rates across strategies."""
    if not repo_stats:
        logger.warning("No repo stats for per-repo chart")
        return

    repos = list(repo_stats.keys())
    n_repos = len(repos)
    n_strats = len(strategies)

    fig, ax = plt.subplots(figsize=(max(12, n_repos * 1.5), 7))

    x = np.arange(n_repos)
    width = 0.8 / n_strats

    for i, strategy in enumerate(strategies):
        rates = [repo_stats[repo].get(strategy, {}).get("rate", 0) * 100 for repo in repos]
        color = STRATEGY_COLORS.get(strategy, "#666666")
        display_name = STRATEGY_NAMES.get(strategy, strategy)
        offset = (i - n_strats / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=display_name, color=color, alpha=0.8)

    # Repo labels with instance counts
    repo_labels = []
    for repo in repos:
        max_n = max(s.get("n", 0) for s in repo_stats[repo].values())
        short_repo = repo.split("/")[-1] if "/" in repo else repo
        repo_labels.append(f"{short_repo}\n(n={max_n})")

    ax.set_xticks(x)
    ax.set_xticklabels(repo_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Patch Production Rate (%)", fontsize=12)
    ax.set_title("Per-Repo Solve Rate by Strategy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-repo chart saved: %s", output_path)


def plot_token_comparison(
    matrix: dict[str, dict[str, dict]],
    strategies: list[str],
    output_path: Path,
) -> None:
    """Box plot of dynamic tokens per strategy across instances."""
    fig, ax = plt.subplots(figsize=(10, 7))

    data = []
    labels = []
    colors = []

    for strategy in strategies:
        if strategy not in matrix:
            continue
        tokens = []
        for iid, result in matrix[strategy].items():
            if not result.get("error") and "dynamic_tokens" in result:
                tokens.append(result["dynamic_tokens"] / 1000)  # K tokens
        if tokens:
            data.append(tokens)
            labels.append(STRATEGY_NAMES.get(strategy, strategy))
            colors.append(STRATEGY_COLORS.get(strategy, "#666666"))

    if not data:
        logger.warning("No token data for token comparison chart")
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Dynamic Tokens (K)", fontsize=12)
    ax.set_title("Token Usage Distribution by Strategy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Token comparison chart saved: %s", output_path)
