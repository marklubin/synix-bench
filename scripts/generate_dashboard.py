#!/usr/bin/env python3
"""Generate HTML tracking dashboards from synix-bench results.

Usage:
    python scripts/generate_dashboard.py [--results-dir results] [--output-dir docs/dashboard]

Produces one HTML page per suite (swebench.html, lens.html) with a
strategy × instance matrix showing pass/fail/not-run status and links
to raw JSON results.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def load_results(results_dir: Path) -> list[dict]:
    """Load all result JSON files from the results directory."""
    results = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            data["_file"] = p.name
            data["_path"] = str(p)
            results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Skipping %s: %s", p.name, e)
    return results


def extract_task_rows(results: list[dict], suite_filter: str) -> list[dict]:
    """Flatten results into per-task rows for a given suite."""
    rows = []
    for result in results:
        if result.get("suite") != suite_filter:
            continue
        strategy = result.get("strategy", "?")
        model = result.get("model", "?")
        result_file = result["_file"]
        for task in result.get("tasks", []):
            task_id = task.get("task_id", "?")
            success = task.get("success", False)
            in_tok = task.get("total_input_tokens", 0)
            out_tok = task.get("total_output_tokens", 0)
            wall_s = task.get("wall_time_s", 0)
            error = task.get("raw_result", {}).get("error")

            # Verification details
            verif = task.get("raw_result", {}).get("verification", {})
            verif_passed = verif.get("passed", False) if verif else False

            # SWE-bench specific
            patch = task.get("raw_result", {}).get("patch", "")
            trace = task.get("raw_result", {}).get("trace", [])
            steps = len(trace) if trace else 0

            # LENS specific
            lens_valid = None
            lens_total = None
            if suite_filter == "lens":
                details = verif.get("details", {})
                lens_valid = details.get("valid_count")
                lens_total = details.get("total_count")

            rows.append({
                "task_id": task_id,
                "strategy": strategy,
                "model": model,
                "success": success,
                "verif_passed": verif_passed,
                "in_tok": in_tok,
                "out_tok": out_tok,
                "wall_s": wall_s,
                "steps": steps,
                "has_patch": bool(patch),
                "error": error,
                "result_file": result_file,
                "lens_valid": lens_valid,
                "lens_total": lens_total,
            })
    return rows


def build_matrix(rows: list[dict]) -> tuple[list[str], list[str], dict]:
    """Build strategy × instance matrix from rows."""
    strategies = sorted(set(r["strategy"] for r in rows))
    instances = sorted(set(r["task_id"] for r in rows))
    matrix: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["strategy"], r["task_id"])
        # Keep the latest result if duplicates
        if key not in matrix or r["wall_s"] > 0:
            matrix[key] = r
    return strategies, instances, matrix


def status_cell(row: dict | None, suite: str) -> str:
    """Render a table cell for one strategy×instance result."""
    if row is None:
        return '<td class="not-run">—</td>'
    if row.get("error"):
        err = html.escape(str(row["error"])[:80])
        return f'<td class="error" title="{err}">ERR</td>'
    if row["verif_passed"]:
        label = "PASS"
        cls = "pass"
    elif row["success"]:
        label = "PATCH"
        cls = "patch"
    else:
        label = "FAIL"
        cls = "fail"

    tok = f'{(row["in_tok"] + row["out_tok"]) / 1000:.0f}K'
    time_s = f'{row["wall_s"]:.0f}s'
    steps = f'{row["steps"]}st' if row["steps"] else ""
    detail = f"{tok} {time_s} {steps}".strip()
    title = html.escape(f'{row["result_file"]} | {detail}')
    return f'<td class="{cls}" title="{title}">{label}<br><small>{detail}</small></td>'


def render_html(
    suite: str,
    strategies: list[str],
    instances: list[str],
    matrix: dict,
    rows: list[dict],
    generated_at: str,
) -> str:
    """Render the full HTML dashboard for one suite."""
    # Summary stats
    total = len(rows)
    passed = sum(1 for r in rows if r["verif_passed"])
    patched = sum(1 for r in rows if r["success"] and not r["verif_passed"])
    failed = sum(1 for r in rows if not r["success"] and not r.get("error"))
    errored = sum(1 for r in rows if r.get("error"))
    total_tok = sum(r["in_tok"] + r["out_tok"] for r in rows)

    # Build table rows
    table_rows = []
    for strategy in strategies:
        cells = []
        for inst in instances:
            row = matrix.get((strategy, inst))
            cells.append(status_cell(row, suite))
        s_rows = [r for r in rows if r["strategy"] == strategy]
        s_pass = sum(1 for r in s_rows if r["verif_passed"])
        s_total = len(s_rows)
        rate = f"{s_pass}/{s_total}" if s_total else "—"
        table_rows.append(
            f'<tr><td class="strategy">{html.escape(strategy)}<br>'
            f'<small>{rate}</small></td>{"".join(cells)}</tr>'
        )

    # Shorten instance IDs for column headers
    short_ids = []
    for inst in instances:
        parts = inst.split("__")
        short = parts[-1] if len(parts) > 1 else inst
        short_ids.append(f'<th class="instance" title="{html.escape(inst)}">'
                         f'{html.escape(short)}</th>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>synix-bench — {suite} dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  .summary {{ display: flex; gap: 24px; margin: 16px 0; flex-wrap: wrap; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 12px 20px; }}
  .stat .label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; }}
  .stat .value {{ font-size: 24px; font-weight: 600; }}
  table {{ border-collapse: collapse; margin: 16px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: center;
            font-size: 13px; }}
  th {{ background: #161b22; color: #8b949e; position: sticky; top: 0; }}
  th.instance {{ writing-mode: vertical-lr; text-orientation: mixed;
                 max-width: 40px; font-size: 11px; }}
  td.strategy {{ text-align: left; font-weight: 600; background: #161b22;
                 position: sticky; left: 0; min-width: 140px; }}
  td.pass {{ background: #1a3a2a; color: #3fb950; font-weight: 600; }}
  td.patch {{ background: #2a2a1a; color: #d29922; }}
  td.fail {{ background: #2a1a1a; color: #f85149; }}
  td.error {{ background: #1a1a2a; color: #8b949e; }}
  td.not-run {{ background: #0d1117; color: #484f58; }}
  td small {{ display: block; font-size: 10px; color: #8b949e; margin-top: 2px; }}
  .legend {{ margin: 12px 0; font-size: 12px; color: #8b949e; }}
  .legend span {{ display: inline-block; padding: 2px 8px; margin-right: 8px;
                  border-radius: 3px; }}
  .ts {{ font-size: 11px; color: #484f58; margin-top: 20px; }}
</style>
</head>
<body>
<h1>{suite} tracking</h1>

<div class="summary">
  <div class="stat"><div class="label">Total runs</div><div class="value">{total}</div></div>
  <div class="stat"><div class="label">Verified pass</div><div class="value" style="color:#3fb950">{passed}</div></div>
  <div class="stat"><div class="label">Patch only</div><div class="value" style="color:#d29922">{patched}</div></div>
  <div class="stat"><div class="label">Failed</div><div class="value" style="color:#f85149">{failed}</div></div>
  <div class="stat"><div class="label">Errors</div><div class="value" style="color:#8b949e">{errored}</div></div>
  <div class="stat"><div class="label">Total tokens</div><div class="value">{total_tok / 1000:.0f}K</div></div>
</div>

<div class="legend">
  <span style="background:#1a3a2a;color:#3fb950">PASS</span> verified
  <span style="background:#2a2a1a;color:#d29922">PATCH</span> produced patch, verification failed
  <span style="background:#2a1a1a;color:#f85149">FAIL</span> no patch
  <span style="background:#1a1a2a;color:#8b949e">ERR</span> error
  <span style="background:#0d1117;color:#484f58">—</span> not run
</div>

<table>
<thead>
<tr>
  <th>Strategy</th>
  {"".join(short_ids)}
</tr>
</thead>
<tbody>
{"".join(table_rows)}
</tbody>
</table>

<p class="ts">Generated {generated_at} | synix-bench</p>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate synix-bench dashboards")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="docs/dashboard", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    log.info("Loaded %d result files", len(results))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    for suite in ("swebench", "lens"):
        rows = extract_task_rows(results, suite)
        if not rows:
            log.info("No %s results, skipping", suite)
            continue

        strategies, instances, matrix = build_matrix(rows)
        html_content = render_html(suite, strategies, instances, matrix, rows, generated_at)

        out_path = output_dir / f"{suite}.html"
        out_path.write_text(html_content)
        log.info("Wrote %s (%d strategies × %d instances, %d runs)", out_path, len(strategies), len(instances), len(rows))

    # Generate index page
    index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>synix-bench dashboards</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }}
  a {{ color: #58a6ff; text-decoration: none; font-size: 20px; }}
  a:hover {{ text-decoration: underline; }}
  .links {{ display: flex; flex-direction: column; gap: 16px; margin-top: 20px; }}
</style>
</head>
<body>
<h1>synix-bench tracking</h1>
<div class="links">
  <a href="swebench.html">SWE-bench dashboard</a>
  <a href="lens.html">LENS dashboard</a>
</div>
<p style="font-size:11px;color:#484f58;margin-top:30px">Generated {generated_at}</p>
</body>
</html>"""
    (output_dir / "index.html").write_text(index)
    log.info("Wrote %s", output_dir / "index.html")


if __name__ == "__main__":
    main()
