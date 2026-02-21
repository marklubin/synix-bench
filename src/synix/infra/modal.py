"""Modal infrastructure provider for synix-bench.

Wraps `modal deploy` via subprocess and polls HTTP health checks.
No `modal` Python package dependency — uses subprocess + urllib only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# Default endpoints — vLLM serves a full ASGI app, base_url is for OpenAI SDK
DEFAULT_VLLM_ENDPOINT = (
    "https://marklubin--synix-vllm-inference-serve.modal.run/v1"
)
DEFAULT_EMBEDDINGS_ENDPOINT = (
    "https://marklubin--synix-embeddings-embeddings-v1-embeddings.modal.run"
)

# App definitions live in this package — deployed via `modal deploy <path>`
_INFRA_DIR = Path(__file__).parent

MODAL_APPS = {
    "vllm": {
        "name": "synix-vllm",
        "module": str(_INFRA_DIR / "modal_vllm.py"),
    },
    "embeddings": {
        "name": "synix-embeddings",
        "module": str(_INFRA_DIR / "modal_embeddings.py"),
    },
}


class ModalProvider:
    """Manage Modal inference endpoints for synix-bench.

    Handles deploy, health-check, and teardown of Modal apps.
    SWE-bench containers still run locally — this only manages the LLM endpoint.
    """

    def __init__(
        self,
        api_token: str | None = None,
        vllm_endpoint: str | None = None,
        embeddings_endpoint: str | None = None,
    ) -> None:
        self.api_token = api_token or ""
        self.vllm_endpoint = vllm_endpoint or DEFAULT_VLLM_ENDPOINT
        self.embeddings_endpoint = embeddings_endpoint or DEFAULT_EMBEDDINGS_ENDPOINT
        self._deployed_apps: list[str] = []

    def deploy(self, config: dict | None = None, timeout: int = 300) -> str:
        """Deploy Modal apps and return the vLLM endpoint URL.

        Args:
            config: Optional config dict (unused for now, reserved for future knobs).
            timeout: Health check timeout in seconds.

        Returns:
            The vLLM endpoint URL ready to receive requests.
        """
        self._deploy_app("vllm")
        if not self._health_check("vllm", timeout):
            raise RuntimeError(
                f"vLLM health check failed after {timeout}s. "
                f"Endpoint: {self.vllm_endpoint}"
            )
        return self.vllm_endpoint

    def teardown(self, endpoint: str | None = None) -> None:
        """Stop deployed Modal apps."""
        for app_key in self._deployed_apps:
            app_name = MODAL_APPS[app_key]["name"]
            log.info("Stopping Modal app: %s", app_name)
            try:
                proc = subprocess.run(
                    ["modal", "app", "stop", app_name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    log.warning(
                        "Failed to stop %s (exit %d): %s",
                        app_name, proc.returncode, proc.stderr,
                    )
                else:
                    log.info("Stopped %s", app_name)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log.warning("Could not stop %s: %s", app_name, e)
        self._deployed_apps.clear()

    def status(self, endpoint: str | None = None) -> dict:
        """Ping endpoints and return status dict."""
        return {
            "vllm": self._ping("vllm", timeout=10),
            "embeddings": self._ping("embeddings", timeout=10),
        }

    def get_endpoint(self) -> str:
        """Return the vLLM endpoint URL."""
        return self.vllm_endpoint

    def _deploy_app(self, name: str) -> None:
        """Run `modal deploy` for a single app (app files live in this package)."""
        app_cfg = MODAL_APPS[name]
        log.info("Deploying Modal app: %s", app_cfg["name"])

        proc = subprocess.run(
            ["modal", "deploy", app_cfg["module"]],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"modal deploy failed for {app_cfg['name']} "
                f"(exit {proc.returncode}): {proc.stderr or proc.stdout}"
            )

        for line in proc.stdout.splitlines():
            if any(kw in line.lower() for kw in ["deployed", "created"]):
                log.info("[deploy] %s", line.strip())

        self._deployed_apps.append(name)

    def _health_check(self, name: str, timeout: int) -> bool:
        """Poll endpoint with retries until healthy or timeout."""
        log.info("Health-checking %s (timeout=%ds)...", name, timeout)
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            elapsed = int(time.monotonic() - start)
            try:
                if name == "vllm":
                    # vLLM serves a full ASGI app — use /health GET endpoint
                    health_url = self.vllm_endpoint.rstrip("/v1") + "/health"
                    req = urllib.request.Request(health_url, method="GET")
                else:
                    # Embeddings — POST with minimal payload
                    endpoint = self._endpoint_for(name)
                    payload = self._health_payload(name)
                    headers = {"Content-Type": "application/json"}
                    if self.api_token:
                        headers["Authorization"] = f"Bearer {self.api_token}"
                    req = urllib.request.Request(
                        endpoint, data=payload, headers=headers, method="POST",
                    )

                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read())

                # Validate response
                if name == "vllm":
                    if body.get("status") == "ok":
                        log.info("[health] [%ds] %s OK!", elapsed, name)
                        return True
                    log.info("[health] [%ds] Unexpected response: %s, retrying...", elapsed, body)
                    time.sleep(2)
                    continue
                elif name == "embeddings":
                    data = body.get("data", [])
                    if not data or not data[0].get("embedding"):
                        log.info("[health] [%ds] No embedding data, retrying...", elapsed)
                        time.sleep(2)
                        continue
                    log.info("[health] [%ds] %s OK!", elapsed, name)
                    return True

            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode()[:200]
                except Exception:
                    pass
                if e.code == 303:
                    log.info("[health] [%ds] Container cold-starting (303)...", elapsed)
                else:
                    log.info("[health] [%ds] HTTP %d: %s", elapsed, e.code, err_body[:100])
                time.sleep(3)

            except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
                log.info("[health] [%ds] Waiting... (%s)", elapsed, type(e).__name__)
                time.sleep(3)

        log.error("[health] %s TIMEOUT after %ds", name, timeout)
        return False

    def _ping(self, name: str, timeout: int = 10) -> bool:
        """Send a minimal request to check if endpoint is up."""
        try:
            if name == "vllm":
                health_url = self.vllm_endpoint.rstrip("/v1") + "/health"
                req = urllib.request.Request(health_url, method="GET")
            else:
                endpoint = self._endpoint_for(name)
                payload = self._health_payload(name)
                headers = {"Content-Type": "application/json"}
                if self.api_token:
                    headers["Authorization"] = f"Bearer {self.api_token}"
                req = urllib.request.Request(
                    endpoint, data=payload, headers=headers, method="POST",
                )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                json.loads(resp.read())
            return True
        except Exception:
            return False

    def _endpoint_for(self, name: str) -> str:
        if name == "vllm":
            return self.vllm_endpoint
        elif name == "embeddings":
            return self.embeddings_endpoint
        else:
            raise ValueError(f"Unknown app: {name}")

    @staticmethod
    def _health_payload(name: str) -> bytes:
        if name == "vllm":
            return json.dumps({
                "messages": [{"role": "user", "content": "Say OK."}],
                "temperature": 0,
                "max_tokens": 8,
            }).encode()
        elif name == "embeddings":
            return json.dumps({
                "input": "health check",
                "model": "Qwen/Qwen3-Embedding-0.6B",
            }).encode()
        else:
            raise ValueError(f"Unknown app: {name}")
