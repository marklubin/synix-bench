"""Modal embeddings app — Qwen3-Embedding-0.6B on T4.

Deployed via `modal deploy src/synix/infra/modal_embeddings.py`.
Copied from synix-modal/src/synix_modal/embeddings.py (validated config).
"""

import os

import fastapi
import modal

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
VOLUME_NAME = "synix-model-cache"
CACHE_DIR = "/root/.cache/huggingface"

embeddings_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers",
        "huggingface_hub[hf_transfer]",
        "fastapi[standard]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

app = modal.App("synix-embeddings")
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _check_auth(authorization: str):
    expected = os.environ.get("SYNIX_API_TOKEN", "")
    token = authorization.removeprefix("Bearer ").strip()
    if not expected or token != expected:
        raise fastapi.HTTPException(status_code=401, detail="Invalid or missing API token")


@app.function(
    image=embeddings_image,
    volumes={CACHE_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=900,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
    model_volume.commit()


@app.cls(
    image=embeddings_image,
    gpu="T4",
    volumes={CACHE_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("synix-api-token")],
    timeout=60,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=64)
class Embeddings:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_ID, cache_folder=CACHE_DIR)

    @modal.fastapi_endpoint(method="POST")
    async def v1_embeddings(
        self,
        request: dict,
        authorization: str = fastapi.Header(alias="Authorization", default=""),
    ):
        _check_auth(authorization)

        raw_input = request.get("input", "")
        if isinstance(raw_input, str):
            texts = [raw_input]
        else:
            texts = list(raw_input)

        vectors = self.model.encode(texts, normalize_embeddings=True)

        data = []
        for i, vec in enumerate(vectors):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": vec.tolist(),
            })

        total_tokens = sum(len(t.split()) for t in texts)

        return {
            "object": "list",
            "data": data,
            "model": request.get("model", MODEL_ID),
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
