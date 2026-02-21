"""Paired bootstrap significance testing.

Extracted from hybrid-memory-bench/scripts/analyze_matrix.py.
Provides paired_bootstrap() and run_all_comparisons() for comparing
strategy performance across shared instances.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def build_matrix(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Build strategy x instance matrix from flat results list.

    Returns {strategy: {instance_id: result_dict}}.
    For multi-trial cells, picks the best trial (has_patch=True preferred,
    then lowest dynamic_tokens).
    """
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        strategy = r.get("layout", r.get("strategy", "unknown"))
        instance_id = r.get("instance_id", r.get("task_id", "unknown"))
        cells[(strategy, instance_id)].append(r)

    matrix: dict[str, dict[str, dict]] = defaultdict(dict)
    for (strategy, instance_id), trials in cells.items():
        if len(trials) == 1:
            matrix[strategy][instance_id] = trials[0]
        else:
            completed = [t for t in trials if not t.get("error")]
            if not completed:
                matrix[strategy][instance_id] = trials[0]
            else:
                patched = [t for t in completed if t.get("has_patch", False)]
                if patched:
                    matrix[strategy][instance_id] = min(
                        patched, key=lambda t: t.get("dynamic_tokens", float("inf"))
                    )
                else:
                    matrix[strategy][instance_id] = min(
                        completed, key=lambda t: t.get("dynamic_tokens", float("inf"))
                    )

    return dict(matrix)


def extract_paired_outcomes(
    matrix: dict[str, dict[str, dict]],
    strategy_a: str,
    strategy_b: str,
    metric: str = "has_patch",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract paired outcomes for two strategies on shared instances.

    Returns (outcomes_a, outcomes_b, instance_ids) where outcomes are arrays.
    Errors count as 0 (failure).
    """
    shared_instances = sorted(
        set(matrix.get(strategy_a, {}).keys()) & set(matrix.get(strategy_b, {}).keys())
    )

    outcomes_a = []
    outcomes_b = []
    for iid in shared_instances:
        ra = matrix[strategy_a][iid]
        rb = matrix[strategy_b][iid]

        if metric == "has_patch":
            va = 1 if (not ra.get("error") and ra.get("has_patch", False)) else 0
            vb = 1 if (not rb.get("error") and rb.get("has_patch", False)) else 0
        elif metric == "verifier_pass":
            va = 1 if (not ra.get("error") and ra.get("verifier_passes", 0) > 0) else 0
            vb = 1 if (not rb.get("error") and rb.get("verifier_passes", 0) > 0) else 0
        elif metric == "dynamic_tokens":
            va = ra.get("dynamic_tokens", 0) if not ra.get("error") else 0
            vb = rb.get("dynamic_tokens", 0) if not rb.get("error") else 0
        elif metric == "exit_context_tokens":
            va = ra.get("exit_context_tokens", 0) if not ra.get("error") else 0
            vb = rb.get("exit_context_tokens", 0) if not rb.get("error") else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        outcomes_a.append(va)
        outcomes_b.append(vb)

    return np.array(outcomes_a), np.array(outcomes_b), shared_instances


def paired_bootstrap(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap test for the difference in means.

    For binary outcomes (pass/fail), this tests the difference in solve rates.
    For continuous outcomes (tokens), this tests the difference in means.

    Returns dict with: observed_diff, ci_low, ci_high, p_value, n_instances.
    """
    rng = np.random.RandomState(seed)
    n = len(outcomes_a)

    # Observed difference: mean(B) - mean(A)
    obs_diff = np.mean(outcomes_b) - np.mean(outcomes_a)

    # Bootstrap: resample paired differences
    diffs = outcomes_b - outcomes_a
    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = np.mean(diffs[idx])

    # Confidence interval
    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    # p-value: proportion of bootstrap samples with sign opposite to observed
    if obs_diff > 0:
        p_value = np.mean(boot_diffs <= 0)
    elif obs_diff < 0:
        p_value = np.mean(boot_diffs >= 0)
    else:
        p_value = 1.0

    # Two-sided p-value
    p_value = min(2 * p_value, 1.0)

    return {
        "observed_diff": float(obs_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_instances": n,
        "mean_a": float(np.mean(outcomes_a)),
        "mean_b": float(np.mean(outcomes_b)),
        "bootstrap_mean": float(np.mean(boot_diffs)),
        "bootstrap_std": float(np.std(boot_diffs)),
    }


def run_all_comparisons(
    matrix: dict[str, dict[str, dict]],
    strategies: list[str],
    control: str,
    metric: str = "has_patch",
    n_bootstrap: int = 10000,
    n_comparisons: int = 5,
) -> list[dict]:
    """Run paired bootstrap comparing each strategy against the control.

    Applies Bonferroni correction for n_comparisons pairwise tests.
    """
    results = []
    alpha_corrected = 0.05 / n_comparisons

    for strategy in strategies:
        if strategy == control:
            continue

        a, b, instances = extract_paired_outcomes(matrix, control, strategy, metric=metric)
        if len(instances) == 0:
            logger.warning("No shared instances for %s vs %s — skipping", control, strategy)
            continue

        boot = paired_bootstrap(a, b, n_bootstrap=n_bootstrap, alpha=alpha_corrected)
        boot["strategy_a"] = control
        boot["strategy_b"] = strategy
        boot["metric"] = metric
        boot["alpha_corrected"] = alpha_corrected
        boot["significant"] = boot["p_value"] < alpha_corrected and (
            boot["ci_low"] > 0 or boot["ci_high"] < 0
        )

        results.append(boot)
        sig_marker = " ***" if boot["significant"] else ""
        logger.info(
            "%s vs %s: diff=%.3f CI=[%.3f, %.3f] p=%.4f n=%d%s",
            control, strategy,
            boot["observed_diff"], boot["ci_low"], boot["ci_high"],
            boot["p_value"], boot["n_instances"], sig_marker,
        )

    return results


def compute_per_repo_stats(
    matrix: dict[str, dict[str, dict]],
    strategies: list[str],
) -> dict[str, dict[str, dict]]:
    """Compute per-repo solve rates for each strategy.

    Returns {repo: {strategy: {rate, n, patches}}}.
    """
    all_instances: set[str] = set()
    for strat_data in matrix.values():
        all_instances.update(strat_data.keys())

    by_repo: dict[str, list[str]] = defaultdict(list)
    for iid in all_instances:
        repo = iid.rsplit("-", 1)[0].replace("__", "/")
        by_repo[repo].append(iid)

    repo_stats: dict[str, dict[str, dict]] = {}
    for repo, instances in sorted(by_repo.items(), key=lambda x: -len(x[1])):
        repo_stats[repo] = {}
        for strategy in strategies:
            n = 0
            patches = 0
            for iid in instances:
                if iid in matrix.get(strategy, {}):
                    r = matrix[strategy][iid]
                    n += 1
                    if not r.get("error") and r.get("has_patch", False):
                        patches += 1
            rate = patches / n if n > 0 else 0
            repo_stats[repo][strategy] = {"rate": rate, "n": n, "patches": patches}

    return repo_stats
