"""Modal vLLM inference app — Qwen3-32B-AWQ on H200.

Deployed via `modal deploy src/synix/infra/modal_vllm.py`.
Serves an OpenAI-compatible API at /v1/chat/completions.

Usage with OpenAI SDK:
    client = OpenAI(
        base_url="https://marklubin--synix-vllm-serve.modal.run/v1",
        api_key=SYNIX_API_TOKEN,
    )
"""

import os
import subprocess
import time

import fastapi
import modal

MODEL_ID = "Qwen/Qwen3-32B-AWQ"
MODEL_REVISION = "main"
VOLUME_NAME = "synix-model-cache"
VLLM_PORT = 8000

CACHE_DIR = "/root/.cache/huggingface"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "vllm==0.13.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

app = modal.App("synix-vllm")
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- FastAPI app with OpenAI-compatible routing ---

web_app = fastapi.FastAPI()


def _check_auth(authorization: str):
    expected = os.environ.get("SYNIX_API_TOKEN", "")
    token = authorization.removeprefix("Bearer ").strip()
    if not expected or token != expected:
        raise fastapi.HTTPException(status_code=401, detail="Invalid or missing API token")


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@web_app.post("/v1/chat/completions")
async def chat_completions(
    request: dict,
    authorization: str = fastapi.Header(alias="Authorization", default=""),
):
    """OpenAI-compatible chat completions endpoint."""
    import json
    import re
    import urllib.request

    _check_auth(authorization)

    # Force thinking off at the server — no client needs to remember
    request.setdefault("chat_template_kwargs", {})
    request["chat_template_kwargs"]["enable_thinking"] = False

    body = json.dumps(request).encode()
    req = urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=280) as resp:
        result = json.loads(resp.read())

    # Strip <think>...</think> blocks from all responses
    for choice in result.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content")
        if content:
            msg["content"] = re.sub(
                r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
            ).strip()

    return result


# --- Modal infrastructure ---


@app.function(
    image=vllm_image,
    volumes={CACHE_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=900,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
    )
    model_volume.commit()


@app.cls(
    image=vllm_image,
    gpu="H200",
    volumes={CACHE_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("synix-api-token")],
    timeout=300,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=16)
class Inference:
    @modal.enter()
    def start_engine(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_ID,
            "--revision", MODEL_REVISION,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.95",
            "--max-num-seqs", "32",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--trust-remote-code",
            "--disable-log-requests",
            "--served-model-name", MODEL_ID, "llm",
        ]
        self.proc = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 300):
        import urllib.error
        import urllib.request

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                req = urllib.request.Request(
                    f"http://localhost:{VLLM_PORT}/health"
                )
                urllib.request.urlopen(req, timeout=2)
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                if self.proc.poll() is not None:
                    output = self.proc.stdout.read().decode() if self.proc.stdout else ""
                    raise RuntimeError(
                        f"vLLM server exited with code {self.proc.returncode}:\n{output}"
                    )
                time.sleep(1)
        raise TimeoutError(
            f"vLLM server did not become ready within {timeout}s"
        )

    @modal.exit()
    def stop_engine(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    @modal.asgi_app()
    def serve(self):
        return web_app
