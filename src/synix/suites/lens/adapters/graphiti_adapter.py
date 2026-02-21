"""Graphiti (temporal knowledge graph) memory adapter for LENS."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

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


class _AsyncRunner:
    """Hosts a persistent event loop in a daemon thread for sync->async bridging."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 600.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


_RUNNER: _AsyncRunner | None = None
_RUNNER_LOCK = threading.Lock()


def _get_runner() -> _AsyncRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = _AsyncRunner()
    return _RUNNER


@register_adapter("graphiti")
class GraphitiAdapter(MemoryAdapter):
    """Graphiti temporal knowledge graph adapter for LENS."""

    requires_metering: bool = False

    def __init__(self) -> None:
        self._llm_api_key = os.environ.get("GRAPHITI_LLM_API_KEY", "")
        self._llm_model = os.environ.get(
            "GRAPHITI_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        self._llm_base_url = os.environ.get(
            "GRAPHITI_LLM_BASE_URL", "https://api.together.xyz/v1"
        )
        self._embed_api_key = os.environ.get("GRAPHITI_EMBED_API_KEY", "")
        self._embed_model = os.environ.get(
            "GRAPHITI_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"
        )
        self._embed_base_url = os.environ.get(
            "GRAPHITI_EMBED_BASE_URL", "https://api.together.xyz/v1"
        )
        self._embed_dim = int(os.environ.get("GRAPHITI_EMBED_DIM", "768"))
        self._falkordb_host = os.environ.get("GRAPHITI_FALKORDB_HOST", "localhost")
        self._falkordb_port = int(os.environ.get("GRAPHITI_FALKORDB_PORT", "6379"))

        self._graphiti = None
        self._db_name: str | None = None
        self._text_cache: dict[str, str] = {}
        self._ep_uuid_to_id: dict[str, str] = {}
        self._pending_episodes: list[dict] = []

    def _make_graphiti(self, db_name: str):
        try:
            from graphiti_core import Graphiti  # noqa: PLC0415
            from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: PLC0415
            from graphiti_core.embedder.openai import (  # noqa: PLC0415
                OpenAIEmbedder,
                OpenAIEmbedderConfig,
            )
            from graphiti_core.llm_client.config import LLMConfig  # noqa: PLC0415
            from graphiti_core.llm_client.openai_generic_client import (  # noqa: PLC0415
                OpenAIGenericClient,
            )
        except ImportError as e:
            raise AdapterError(
                "graphiti-core[falkordb] not installed. Run: uv add graphiti-core[falkordb]"
            ) from e

        llm_client = OpenAIGenericClient(
            LLMConfig(
                api_key=self._llm_api_key,
                model=self._llm_model,
                base_url=self._llm_base_url,
            )
        )

        embed_dim = self._embed_dim
        embed_model = self._embed_model
        embed_api_key = self._embed_api_key
        embed_base_url = self._embed_base_url

        class _ChunkedEmbedder(OpenAIEmbedder):
            _BATCH = 20

            async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
                results: list[list[float]] = []
                for i in range(0, len(input_data_list), self._BATCH):
                    chunk = input_data_list[i : i + self._BATCH]
                    chunk_results = await super().create_batch(chunk)
                    results.extend(chunk_results)
                return results

        embedder = _ChunkedEmbedder(
            OpenAIEmbedderConfig(
                embedding_model=embed_model,
                api_key=embed_api_key,
                base_url=embed_base_url,
                embedding_dim=embed_dim,
            )
        )
        driver = FalkorDriver(
            host=self._falkordb_host,
            port=self._falkordb_port,
            database=db_name,
        )
        return Graphiti(graph_driver=driver, llm_client=llm_client, embedder=embedder)

    def reset(self, scope_id: str, cache_key: str | None = None) -> None:
        suffix = cache_key or uuid.uuid4().hex[:8]
        safe_scope = "".join(c if c.isalnum() or c == "_" else "_" for c in scope_id)
        self._db_name = f"synix_{safe_scope}_{suffix}"
        self._text_cache = {}
        self._ep_uuid_to_id = {}
        self._pending_episodes = []

        try:
            self._graphiti = self._make_graphiti(self._db_name)
            _get_runner().run(
                self._graphiti.build_indices_and_constraints(), timeout=60.0
            )
        except AdapterError:
            raise
        except Exception as e:
            raise AdapterError(
                f"Failed to initialize Graphiti with FalkorDB at "
                f"{self._falkordb_host}:{self._falkordb_port} (db={self._db_name!r}). "
                f"Is FalkorDB running? Error: {e}"
            ) from e

    def get_cache_state(self) -> dict | None:
        if not self._db_name:
            return None
        return {
            "db_name": self._db_name,
            "text_cache": self._text_cache,
            "ep_uuid_to_id": self._ep_uuid_to_id,
        }

    def restore_cache_state(self, state: dict) -> bool:
        try:
            db_name = state["db_name"]
            self._graphiti = self._make_graphiti(db_name)
            self._db_name = db_name
            self._text_cache = state.get("text_cache", {})
            self._ep_uuid_to_id = state.get("ep_uuid_to_id", {})
            self._pending_episodes = []
            log.info("Restored Graphiti cache: db=%s, %d episodes", db_name, len(self._text_cache))
            return True
        except Exception as e:
            log.warning("Failed to restore Graphiti cache: %s", e)
            return False

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        if not self._graphiti:
            raise AdapterError("reset() must be called before ingest()")

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        content = f"[{episode_id}] {timestamp}: {text}"
        self._pending_episodes.append(
            {"episode_id": episode_id, "content": content, "timestamp": ts}
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._pending_episodes or not self._graphiti:
            return

        try:
            from graphiti_core.nodes import EpisodeType  # noqa: PLC0415
        except ImportError as e:
            raise AdapterError("graphiti-core not installed") from e

        graphiti = self._graphiti
        pending = list(self._pending_episodes)

        async def _add_batch():
            sem = asyncio.Semaphore(8)

            async def _add_one(item):
                async with sem:
                    result = await graphiti.add_episode(
                        name=item["episode_id"],
                        episode_body=item["content"],
                        source_description="LENS longitudinal operational log episode",
                        reference_time=item["timestamp"],
                        source=EpisodeType.text,
                    )
                    return item["episode_id"], result

            return await asyncio.gather(
                *[_add_one(i) for i in pending], return_exceptions=True
            )

        timeout = 60.0 * len(pending)
        results = _get_runner().run(_add_batch(), timeout=timeout)

        errors = []
        for i, result in enumerate(results):
            episode_id = pending[i]["episode_id"]
            if isinstance(result, BaseException):
                log.error(
                    "add_episode failed for %r at checkpoint %d: %s",
                    episode_id, checkpoint, result,
                )
                errors.append((episode_id, result))
                continue

            _, ep_result = result
            if ep_result and ep_result.episode:
                ep_uuid = str(ep_result.episode.uuid)
                self._ep_uuid_to_id[ep_uuid] = episode_id
            else:
                log.warning(
                    "add_episode returned no episode node for %r", episode_id
                )

        self._pending_episodes = []

        if errors:
            failed_ids = [eid for eid, _ in errors]
            raise AdapterError(
                f"Graphiti add_episode failed for {len(errors)} episode(s) "
                f"at checkpoint {checkpoint}: {failed_ids}"
            )

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip() or not self._graphiti:
            return []

        cap = limit or 10
        try:
            from graphiti_core.search.search_config_recipes import (  # noqa: PLC0415
                EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
            )
            results = _get_runner().run(
                self._graphiti._search(query, config=EDGE_HYBRID_SEARCH_EPISODE_MENTIONS),
                timeout=60.0,
            )
            edges = results.edges or []
        except Exception as e:
            log.warning("Graphiti _search failed: %s", e)
            return []

        search_results: list[SearchResult] = []
        seen_episode_ids: set[str] = set()

        for edge in edges:
            if len(search_results) >= cap:
                break

            ep_id: str | None = None
            ep_uuids = getattr(edge, "episodes", None) or []
            for ep_uuid in ep_uuids:
                candidate = self._ep_uuid_to_id.get(str(ep_uuid))
                if candidate:
                    ep_id = candidate
                    break

            if ep_id is None:
                continue

            if ep_id in seen_episode_ids:
                continue
            seen_episode_ids.add(ep_id)

            fact = getattr(edge, "fact", "") or ""
            search_results.append(
                SearchResult(
                    ref_id=ep_id,
                    text=fact[:500],
                    score=0.5,
                    metadata={"edge_uuid": str(getattr(edge, "uuid", ""))},
                )
            )

        return search_results

    def retrieve(self, ref_id: str) -> Document | None:
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "graph", "keyword"],
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
