"""HTTP server for the stack trace viewer.

Serves viewer.html and provides a JSON index of trace files.

Usage:
    python -m synix.viewer.server [--port 8787]

Endpoints:
    /              -> viewer.html
    /traces        -> JSON index of all *stack*.json files
    /traces/<file> -> individual trace JSON
"""

from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Look for results in synix-bench/results/ by default
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"
VIEWER_PATH = Path(__file__).resolve().parent / "stack_viewer.html"


def build_index() -> list[dict]:
    """Scan results/ for stack trace files and extract meta."""
    if not RESULTS_DIR.exists():
        return []
    traces = []
    for f in sorted(RESULTS_DIR.glob("*stack*.json"), reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            meta = data.get("meta", {})
            traces.append({
                "file": f.name,
                "name": meta.get("name", ""),
                "task": (meta.get("task") or "")[:200],
                "model": meta.get("model", ""),
                "steps": meta.get("total_steps", 0),
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "managed_tokens": meta.get("managed_tokens"),
                "cached_tokens": meta.get("cached_tokens"),
                "dynamic_tokens": meta.get("dynamic_tokens"),
                "elapsed_s": meta.get("elapsed_s", 0),
                "max_depth": meta.get("max_depth", 0),
                "timestamp": meta.get("timestamp", ""),
                "result": (meta.get("result") or "")[:200],
                "reg_total_updates": meta.get("reg_total_updates"),
                "peak_reg_bytes": meta.get("peak_reg_bytes"),
                "num_regs": max(
                    (sum(1 for v in s.get("registers", {}).values() if v)
                     for s in data.get("steps", [{}])),
                    default=0,
                ) or None,
                "probe_count": meta.get("probe_count", 0),
                "search_stats": meta.get("search_stats"),
            })
        except Exception as e:
            print(f"  skip {f.name}: {e}", file=sys.stderr)
    return traces


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            if VIEWER_PATH.exists():
                self._serve_file(VIEWER_PATH, "text/html")
            else:
                self.send_error(404, "viewer.html not found")
        elif path == "/traces":
            idx = build_index()
            self._serve_json(idx)
        elif path.startswith("/traces/"):
            fname = os.path.basename(path[8:])
            fpath = RESULTS_DIR / fname
            if fpath.exists() and fpath.suffix == ".json":
                self._serve_file(fpath, "application/json")
            else:
                self.send_error(404, f"Not found: {fname}")
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if "/traces" in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Serve stack trace viewer")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory path")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    idx = build_index()
    print(f"Found {len(idx)} stack traces in {RESULTS_DIR}")
    for t in idx[:5]:
        regs = f" regs={t['reg_total_updates']}" if t['reg_total_updates'] is not None else ""
        print(f"  {t['file']}: {t['steps']} steps, {t['input_tokens']:,} tok, {t['elapsed_s']}s{regs}")
    if len(idx) > 5:
        print(f"  ... and {len(idx)-5} more")

    server = HTTPServer((args.host, args.port), Handler)
    print(f"\nServing on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
