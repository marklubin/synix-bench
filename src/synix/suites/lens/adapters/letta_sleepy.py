"""Letta with checkpoint sleep consolidation for LENS."""
from __future__ import annotations

import os
import re

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

from synix.suites.lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from synix.suites.lens.adapters.registry import register_adapter
from synix.core.errors import AdapterError

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")

_PERSONA = (
    "I am an archival memory store for the LENS benchmark. "
    "My role is to accurately store and retrieve sequential episode logs "
    "for longitudinal analysis. I do not editorialize or add context."
)

_DEFAULT_LLM = "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
_DEFAULT_EMBED = "together-oai/text-embedding-3-small"

_SLEEP_SYSTEM = (
    "You are a memory consolidation agent. You have been given a set of "
    "sequential operational episode logs stored in a memory system. "
    "Your task is to analyse these episodes and produce a structured synthesis "
    "that will help answer future analytical questions about the data. "
    "Be specific and cite episode IDs (e.g. [ep_001]) whenever you reference "
    "particular data points."
)

_SLEEP_VARIANTS: dict[int, str | None] = {
    0: None,
    1: (
        "Write a comprehensive summary of all stored episodes. "
        "Include the key information from each episode and cite episode IDs."
    ),
    2: (
        "Organise this information for efficient future retrieval. "
        "Prioritise information that reveals patterns, anomalies, or changes "
        "over time. De-emphasise stable baselines and routine readings that are "
        "unlikely to distinguish one episode from another. "
        "Write a structured synthesis focusing on what is most likely to be "
        "analytically significant."
    ),
    3: (
        "Identify what changed over time, when transitions occurred, "
        "and what correlations exist between different components or metrics. "
        "Focus on cause and effect, progression, and turning points rather than "
        "describing the contents of each individual episode. "
        "Which episodes mark significant state changes, and what drove them?"
    ),
}

_SLEEP_USER_TMPL = """\
You have {n} episodes in memory spanning {first_ts} to {last_ts}.

EPISODES:
{passages}

CONSOLIDATION OBJECTIVE:
{objective}

Write your synthesis below. Cite episode IDs (e.g. [ep_001]) when referencing specific data points.

SYNTHESIS:"""

_MAX_PASSAGE_CHARS = 1200
_MAX_SYNTHESIS_CHARS = 3000


def _parse_ep_id(content: str) -> str:
    m = _EP_ID_RE.match(content)
    return m.group(1) if m else content[:32]


def _strip_provider_prefix(letta_model: str) -> str:
    if "/" in letta_model:
        return letta_model.split("/", 1)[1]
    return letta_model


@register_adapter("letta-sleepy")
class LettaSleepyAdapter(MemoryAdapter):
    """Letta archival memory adapter with checkpoint sleep consolidation."""

    requires_metering: bool = False

    def __init__(self) -> None:
        self._base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._llm_model = os.environ.get("LETTA_LLM_MODEL", _DEFAULT_LLM)
        self._embed_model = os.environ.get("LETTA_EMBED_MODEL", _DEFAULT_EMBED)
        self._variant = int(os.environ.get("LETTA_SLEEP_VARIANT", "2"))
        self._client = None

        self._agent_id: str | None = None
        self._scope_id: str | None = None
        self._text_cache: dict[str, str] = {}
        self._synthesis: str = ""

    def _get_client(self):
        if self._client is None:
            try:
                from letta_client import Letta  # noqa: PLC0415
            except ImportError as e:
                raise AdapterError(
                    "letta-client not installed. Run: pip install letta-client"
                ) from e
            self._client = Letta(base_url=self._base_url, api_key="dummy")
        return self._client

    def _fetch_passages(self) -> list[str]:
        client = self._get_client()
        try:
            page = client.agents.passages.list(agent_id=self._agent_id, limit=500)
            return [p.text for p in page if getattr(p, "text", None)]
        except Exception:
            return []

    def _run_sleep_cycle(self, passages: list[str], checkpoint: int) -> str:
        objective = _SLEEP_VARIANTS.get(self._variant)
        if not objective or not passages:
            return ""

        passage_lines = []
        for p in passages:
            passage_lines.append(p[:_MAX_PASSAGE_CHARS])
        passage_block = "\n\n".join(passage_lines)

        def _ts(text: str) -> str:
            parts = text.split(" ", 2)
            return parts[1].rstrip(":") if len(parts) >= 2 else "unknown"

        first_ts = _ts(passages[-1])
        last_ts = _ts(passages[0])

        user_msg = _SLEEP_USER_TMPL.format(
            n=len(passages),
            first_ts=first_ts,
            last_ts=last_ts,
            passages=passage_block,
            objective=objective,
        )

        api_key = (
            os.environ.get("SYNIX_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        base_url = (
            os.environ.get("SYNIX_LLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model = _strip_provider_prefix(self._llm_model)

        if _OpenAI is None:
            raise AdapterError(
                "openai package required for sleep cycle. Run: pip install openai"
            )

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        oai = _OpenAI(**client_kwargs)
        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SLEEP_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            synthesis = resp.choices[0].message.content or ""
        except Exception:
            return ""

        return synthesis[:_MAX_SYNTHESIS_CHARS]

    def reset(self, scope_id: str) -> None:
        client = self._get_client()
        self._synthesis = ""

        if self._agent_id is not None:
            try:
                client.agents.delete(agent_id=self._agent_id)
            except Exception:
                pass

        try:
            for agent in client.agents.list():
                if agent.name == f"lens-sleepy-{scope_id}":
                    try:
                        client.agents.delete(agent_id=agent.id)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            agent = client.agents.create(
                name=f"lens-sleepy-{scope_id}",
                model=self._llm_model,
                embedding=self._embed_model,
                memory_blocks=[
                    {
                        "label": "human",
                        "value": f"LENS benchmark scope: {scope_id}",
                    },
                    {
                        "label": "persona",
                        "value": _PERSONA,
                    },
                ],
            )
        except Exception as e:
            raise AdapterError(
                f"Failed to create Letta agent. Is the server running at "
                f"{self._base_url}? Error: {e}"
            ) from e

        self._agent_id = agent.id
        self._scope_id = scope_id
        self._text_cache = {}

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        if not self._agent_id:
            raise AdapterError("reset() must be called before ingest()")

        content = f"[{episode_id}] {timestamp}: {text}"
        client = self._get_client()
        client.agents.passages.create(
            agent_id=self._agent_id,
            text=content,
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if self._variant == 0 or not self._agent_id:
            return
        passages = self._fetch_passages()
        if not passages:
            return
        self._synthesis = self._run_sleep_cycle(passages, checkpoint)

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if not self._agent_id:
            return []

        cap = limit or 10
        client = self._get_client()

        passage_cap = cap - 1 if (self._synthesis and cap > 1) else cap

        try:
            response = client.agents.passages.search(
                agent_id=self._agent_id,
                query=query,
            )
        except Exception:
            response = None

        raw = getattr(response, "results", None) if response else None
        if raw is None:
            raw = response if isinstance(response, list) else []

        results: list[SearchResult] = []

        if self._synthesis:
            results.append(SearchResult(
                ref_id="synthesis",
                text=self._synthesis,
                score=0.5,
                metadata={"type": "consolidated_synthesis"},
            ))

        for item in raw[:passage_cap]:
            content = getattr(item, "content", None) or getattr(item, "text", "")
            ep_id = _parse_ep_id(content)
            results.append(SearchResult(
                ref_id=ep_id,
                text=content[:500],
                score=0.0,
                metadata={"timestamp": getattr(item, "timestamp", "")},
            ))

        return results

    def retrieve(self, ref_id: str) -> Document | None:
        if ref_id == "synthesis":
            if self._synthesis:
                return Document(ref_id="synthesis", text=self._synthesis)
            return None
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", f"sleep-consolidated-v{self._variant}"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple full episodes by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times — it uses only "
                        "one tool call instead of one per document. "
                        "After memory_search, pass all ref_ids you want to read to this tool. "
                        "Note: ref_id='synthesis' returns the pre-consolidated memory synthesis."
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
