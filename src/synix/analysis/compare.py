"""Cross-run comparison and result loading.

Ported from hybrid-memory-bench/src/hybrid_memory/compare.py.
Loads result JSON files and prints comparison tables grouped by
condition or task_id.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_results(results_dir: str = "results", run_ids: list[str] | None = None) -> list[dict]:
    """Load result JSON files from the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning("Results directory does not exist: %s", results_dir)
        return []

    results = []
    for f in sorted(results_path.glob("*.json")):
        if run_ids and f.stem not in run_ids:
            continue
        try:
            data = json.loads(f.read_text())
            results.append(data)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed result file: %s", f)

    logger.info("Loaded %d results", len(results))
    return results


def compare(results: list[dict], group_by: str = "condition") -> None:
    """Print comparison table grouped by condition or task_id."""
    if not results:
        print("No results to compare.")
        return

    # Find naive baseline for cost ratio calculation
    naive_tokens: dict[str, int] = {}
    for r in results:
        config = r.get("config", {})
        if config.get("condition") == "naive":
            task_id = config.get("task_id", "")
            naive_tokens[task_id] = r.get("metrics", {}).get("total_frontier_input_tokens", 0)

    if group_by == "condition":
        _compare_by_condition(results, naive_tokens)
    elif group_by == "task_id":
        _compare_by_task(results, naive_tokens)
    else:
        print(f"Unknown group_by: {group_by}. Use 'condition' or 'task_id'.")


def _compare_by_condition(results: list[dict], naive_tokens: dict[str, int]) -> None:
    """Aggregate metrics by condition across all tasks."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        condition = r.get("config", {}).get("condition", "unknown")
        groups[condition].append(r)

    print()
    print(f"{'Condition':<15} {'Tasks':>5} {'Avg Score':>10} {'Avg Prompt Tok':>15} {'Avg Manager Tok':>16} {'Cost Ratio':>11}")
    print("-" * 78)

    for condition in ["naive", "masking", "managed", "hybrid", "tiered"]:
        if condition not in groups:
            continue
        group = groups[condition]
        n_tasks = len(group)
        avg_score = _avg([r.get("score") for r in group if r.get("score") is not None])
        avg_prompt = _avg([r.get("metrics", {}).get("total_frontier_input_tokens", 0) for r in group])
        avg_manager = _avg([
            r.get("metrics", {}).get("total_manager_input_tokens", 0)
            + r.get("metrics", {}).get("total_manager_output_tokens", 0)
            for r in group
        ])

        # Cost ratio vs naive
        cost_ratios = []
        for r in group:
            task_id = r.get("config", {}).get("task_id", "")
            if task_id in naive_tokens and naive_tokens[task_id] > 0:
                ratio = r.get("metrics", {}).get("total_frontier_input_tokens", 0) / naive_tokens[task_id]
                cost_ratios.append(ratio)
        avg_cost_ratio = _avg(cost_ratios) if cost_ratios else None

        score_str = f"{avg_score:.1f}" if avg_score is not None else "pending"
        prompt_str = f"{avg_prompt:,.0f}" if avg_prompt is not None else "-"
        manager_str = f"{avg_manager:,.0f}" if avg_manager is not None else "-"
        ratio_str = f"{avg_cost_ratio:.2f}x" if avg_cost_ratio is not None else "-"

        print(f"{condition:<15} {n_tasks:>5} {score_str:>10} {prompt_str:>15} {manager_str:>16} {ratio_str:>11}")

    print()


def _compare_by_task(results: list[dict], naive_tokens: dict[str, int]) -> None:
    """Print per-task breakdown tables."""
    tasks: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        task_id = r.get("config", {}).get("task_id", "unknown")
        tasks[task_id].append(r)

    for task_id, task_results in sorted(tasks.items()):
        n_steps = task_results[0].get("metrics", {}).get("total_steps", "?")
        print(f"\nTask: {task_id} ({n_steps} steps)")
        print(f"{'Condition':<15} {'Score':>8} {'Prompt Tok':>12} {'Manager Tok':>12} {'Cost Ratio':>11}")
        print("-" * 62)

        for r in sorted(task_results, key=lambda x: x.get("config", {}).get("condition", "")):
            condition = r.get("config", {}).get("condition", "unknown")
            score = r.get("score")
            prompt_tok = r.get("metrics", {}).get("total_frontier_input_tokens", 0)
            manager_tok = (
                r.get("metrics", {}).get("total_manager_input_tokens", 0)
                + r.get("metrics", {}).get("total_manager_output_tokens", 0)
            )

            cost_ratio = None
            if task_id in naive_tokens and naive_tokens[task_id] > 0:
                cost_ratio = prompt_tok / naive_tokens[task_id]

            score_str = f"{score:.1f}" if score is not None else "pending"
            ratio_str = f"{cost_ratio:.2f}x" if cost_ratio is not None else "-"

            print(f"{condition:<15} {score_str:>8} {prompt_tok:>12,} {manager_tok:>12,} {ratio_str:>11}")

    print()


def _avg(values: list) -> float | None:
    """Average of non-None values."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None
