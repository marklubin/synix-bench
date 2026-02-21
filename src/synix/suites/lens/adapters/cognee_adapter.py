"""Cognee (GraphRAG) memory adapter for LENS."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import uuid

from synix.suites.lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from synix.suites.lens.adapters.registry import register_adapter
from synix.core.errors import AdapterError

log = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")


def _parse_ep_id(text: str) -> str | None:
    m = _EP_ID_RE.match(text)
    return m.group(1) if m else None


class _AsyncRunner:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 600.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


_RUNNER: _AsyncRunner | None = None
_RUNNER_LOCK = threading.Lock()
_COGNEE_PATCHED = False


def _get_runner() -> _AsyncRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = _AsyncRunner()
    return _RUNNER


@register_adapter("cognee")
class CogneeAdapter(MemoryAdapter):
    """Cognee GraphRAG adapter for LENS."""

    requires_metering: bool = False

    def __init__(self) -> None:
        self._llm_api_key = os.environ.get("COGNEE_LLM_API_KEY", "")
        self._llm_model = os.environ.get(
            "COGNEE_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        self._llm_endpoint = os.environ.get(
            "COGNEE_LLM_ENDPOINT", "https://api.together.xyz/v1"
        )
        self._embed_api_key = os.environ.get("COGNEE_EMBED_API_KEY", "")
        self._embed_model = os.environ.get(
            "COGNEE_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"
        )
        self._embed_endpoint = os.environ.get(
            "COGNEE_EMBED_ENDPOINT", "https://api.together.xyz/v1"
        )
        self._embed_dims = int(os.environ.get("COGNEE_EMBED_DIMS", "768"))

        self._apply_env_config()

        self._dataset_id: str | None = None
        self._text_cache: dict[str, str] = {}
        self._pending_episodes: list[dict] = []

    def _apply_env_config(self) -> None:
        if self._embed_api_key:
            os.environ.setdefault("EMBEDDING_API_KEY", self._embed_api_key)
        if self._embed_model:
            embed_model = self._embed_model
            if not embed_model.startswith(("openai/", "together_ai/", "bedrock/")):
                embed_model = f"together_ai/{embed_model}"
            os.environ.setdefault("EMBEDDING_MODEL", embed_model)
        os.environ.setdefault("EMBEDDING_DIMENSIONS", str(self._embed_dims))
        if self._embed_endpoint:
            os.environ.setdefault("EMBEDDING_ENDPOINT", self._embed_endpoint)
        os.environ.setdefault("EMBEDDING_PROVIDER", "openai")

        try:
            import tiktoken  # noqa: PLC0415

            _orig = tiktoken.encoding_for_model

            def _patched_encoding_for_model(model_name: str):
                try:
                    return _orig(model_name)
                except KeyError:
                    return tiktoken.get_encoding("cl100k_base")

            tiktoken.encoding_for_model = _patched_encoding_for_model
        except Exception:
            pass

    def _get_cognee(self):
        global _COGNEE_PATCHED  # noqa: PLW0603

        try:
            import cognee  # noqa: PLC0415
        except ImportError as e:
            raise AdapterError(
                "cognee not installed. Run: uv add cognee"
            ) from e

        if not _COGNEE_PATCHED:
            try:
                import litellm as _litellm  # noqa: PLC0415

                _orig_aembedding = _litellm.aembedding

                async def _aembedding_no_dims(*args, **kwargs):
                    kwargs.pop("dimensions", None)
                    return await _orig_aembedding(*args, **kwargs)

                _litellm.aembedding = _aembedding_no_dims

                _orig_acompletion = _litellm.acompletion

                async def _acompletion_with_max_tokens(*args, **kwargs):
                    if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
                        kwargs["max_tokens"] = 16384
                    return await _orig_acompletion(*args, **kwargs)

                _litellm.acompletion = _acompletion_with_max_tokens

                from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (  # noqa: PLC0415
                    create_embedding_engine as _cee,
                )

                _cee.cache_clear()
                log.debug("Patched litellm: suppressed dimensions, injected max_tokens=16384")
            except Exception as _pe:
                log.warning("Failed to patch litellm: %s", _pe)
            _COGNEE_PATCHED = True

        llm_model = self._llm_model
        if not llm_model.startswith(("openai/", "together_ai/", "anthropic/", "bedrock/")):
            llm_model = f"openai/{llm_model}"
        try:
            cognee.config.set_llm_config({
                "llm_provider": "openai",
                "llm_model": llm_model,
                "llm_endpoint": self._llm_endpoint,
                "llm_api_key": self._llm_api_key,
            })
        except Exception as e:
            log.warning("cognee.config.set_llm_config failed: %s", e)

        return cognee

    def reset(self, scope_id: str, cache_key: str | None = None) -> None:
        suffix = cache_key or uuid.uuid4().hex[:8]
        safe_scope = "".join(c if c.isalnum() or c == "_" else "_" for c in scope_id)
        self._dataset_id = f"synix_{safe_scope}_{suffix}"
        self._text_cache = {}
        self._pending_episodes = []

        cognee = self._get_cognee()
        runner = _get_runner()

        import shutil  # noqa: PLC0415

        cognee_db_dir = os.path.join(
            os.path.dirname(cognee.__file__),
            ".cognee_system",
            "databases",
        )
        if os.path.isdir(cognee_db_dir):
            try:
                shutil.rmtree(cognee_db_dir)
                log.debug("Removed cognee database directory: %s", cognee_db_dir)
            except OSError as e:
                log.warning("Failed to remove cognee DB dir: %s", e)

        cognee_cache_dir = os.path.join(
            os.path.dirname(cognee.__file__),
            ".cognee_cache",
        )
        if os.path.isdir(cognee_cache_dir):
            try:
                shutil.rmtree(cognee_cache_dir)
            except OSError as e:
                log.warning("Failed to remove cognee cache dir: %s", e)

        try:
            runner.run(cognee.prune.prune_data(), timeout=120.0)
            runner.run(
                cognee.prune.prune_system(
                    graph=True, vector=True, metadata=True, cache=True
                ),
                timeout=120.0,
            )
        except Exception as e:
            log.warning(
                "cognee.prune failed during reset for scope '%s': %s (continuing with fresh dataset)",
                scope_id,
                e,
            )

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        if self._dataset_id is None:
            raise AdapterError("reset() must be called before ingest()")

        content = f"[{episode_id}] {timestamp}: {text}"
        self._pending_episodes.append(
            {"episode_id": episode_id, "content": content}
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._pending_episodes or self._dataset_id is None:
            return

        cognee = self._get_cognee()

        for item in self._pending_episodes:
            try:
                _get_runner().run(
                    cognee.add(
                        data=item["content"],
                        dataset_name=self._dataset_id,
                    ),
                    timeout=120.0,
                )
            except Exception as e:
                raise AdapterError(
                    f"cognee.add() failed for episode '{item['episode_id']}' "
                    f"at checkpoint {checkpoint}: {e}"
                ) from e

        try:
            _get_runner().run(
                cognee.cognify(datasets=[self._dataset_id]),
                timeout=600.0,
            )
        except Exception as e:
            log.warning(
                "cognee.cognify() failed at checkpoint %d (non-fatal, "
                "earlier chunks may still be searchable): %s",
                checkpoint,
                e,
            )

        self._pending_episodes = []

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip() or self._dataset_id is None:
            return []

        cap = limit or 10
        cognee = self._get_cognee()
        SearchType = cognee.SearchType

        raw = None
        for search_type in (SearchType.CHUNKS, SearchType.SUMMARIES):
            try:
                raw = _get_runner().run(
                    cognee.search(
                        query_text=query,
                        query_type=search_type,
                        top_k=cap * 2,
                    ),
                    timeout=60.0,
                )
                st_name = getattr(search_type, "name", str(search_type))
                log.debug(
                    "cognee.search(%s) returned %d items",
                    st_name,
                    len(raw) if raw else 0,
                )
                if raw:
                    break
            except Exception as e:
                st_name = getattr(search_type, "name", str(search_type))
                log.warning("cognee.search(%s) failed: %s", st_name, e)
                continue

        if not raw:
            log.warning(
                "cognee search returned empty for query=%r (tried CHUNKS + SUMMARIES)",
                query[:80],
            )
            return []

        search_results: list[SearchResult] = []
        seen_episode_ids: set[str] = set()

        for item in raw or []:
            if len(search_results) >= cap:
                break

            chunk_list = self._extract_chunks(item)
            for chunk_text in chunk_list:
                if len(search_results) >= cap:
                    break
                if not chunk_text:
                    continue

                ep_id = _parse_ep_id(chunk_text)
                if ep_id is None:
                    ep_id = self._match_episode(chunk_text)

                if ep_id is None or ep_id in seen_episode_ids:
                    continue
                seen_episode_ids.add(ep_id)

                search_results.append(
                    SearchResult(
                        ref_id=ep_id,
                        text=chunk_text[:500],
                        score=0.5,
                    )
                )

        return search_results

    def _extract_chunks(self, item) -> list[str]:
        search_result = getattr(item, "search_result", item)

        if isinstance(search_result, dict) and "search_result" in search_result:
            search_result = search_result["search_result"]

        if isinstance(search_result, dict):
            text = search_result.get("text") or search_result.get("content") or ""
            return [str(text)] if text else []

        if isinstance(search_result, str):
            return [search_result] if search_result else []

        if isinstance(search_result, list):
            texts = []
            for entry in search_result:
                if isinstance(entry, str):
                    texts.append(entry)
                elif isinstance(entry, dict):
                    text = entry.get("text") or entry.get("content") or ""
                    if text:
                        texts.append(str(text))
            return texts

        log.warning(
            "_extract_chunks: unhandled type=%s repr=%.200s",
            type(search_result).__name__,
            repr(search_result),
        )
        return []

    def _match_episode(self, chunk_text: str) -> str | None:
        chunk_lower = chunk_text.lower()[:200]
        for ep_id, ep_text in self._text_cache.items():
            ep_prefix = ep_text[:100].lower()
            if ep_prefix and ep_prefix[:50] in chunk_lower:
                return ep_id
        return None

    def retrieve(self, ref_id: str) -> Document | None:
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "graph"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple full episodes by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times — it uses only "
                        "one tool call instead of one per document. "
                        "After memory_search, pass all ref_ids you want to read to this tool."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference IDs to retrieve.",
                            },
                        },
                        "required": ["ref_ids"],
                    },
                ),
            ],
        )

    def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
        if tool_name == "batch_retrieve":
            ref_ids = arguments.get("ref_ids", [])
            docs = []
            for ref_id in ref_ids:
                doc = self.retrieve(ref_id)
                if doc is not None:
                    docs.append(doc.to_dict())
            return {"documents": docs, "count": len(docs)}
        return super().call_extended_tool(tool_name, arguments)

    def get_cache_state(self) -> dict | None:
        if not self._dataset_id:
            return None
        return {
            "dataset_id": self._dataset_id,
            "text_cache": self._text_cache,
        }

    def restore_cache_state(self, state: dict) -> bool:
        try:
            self._dataset_id = state["dataset_id"]
            self._text_cache = state.get("text_cache", {})
            self._pending_episodes = []
            self._get_cognee()
            log.info(
                "Restored Cognee cache: dataset=%s, %d episodes",
                self._dataset_id,
                len(self._text_cache),
            )
            return True
        except Exception as e:
            log.warning("Failed to restore Cognee cache: %s", e)
            return False
