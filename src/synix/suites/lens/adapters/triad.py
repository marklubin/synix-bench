"""Triad Memory Protocol adapters for LENS.

Six variants of a multi-agent memory system across two facet decompositions:

3-facet (original):
- triad-monolith: Single agent, single notebook (Model 0 baseline)
- triad-panel: 3 parallel specialist agents (Model A)
- triad-conversation: 3 sequential specialist agents (Model B)

4-facet (causation split into event + cause):
- triad4-monolith: Single agent, single notebook
- triad4-panel: 4 parallel specialist agents
- triad4-conversation: 4 sequential specialist agents
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

from synix.suites.lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)
from synix.suites.lens.adapters.registry import register_adapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Facet definitions
# ---------------------------------------------------------------------------

FACETS = ("identity", "relation", "causation")

FACETS_4 = ("entity", "relation", "event", "cause")

_FACET_DESCRIPTIONS = {
    # 3-facet
    "identity": "entity identities, traits, roles, and stable attributes",
    "relation": "relationships, social connections, and interaction patterns between entities",
    "causation": "causal chains, cause-and-effect patterns, and temporal sequences of events",
    "monolith": "all aspects: identities, relationships, and causal patterns",
    # 4-facet
    "entity": (
        "entity identities — what things exist, what makes each one distinct, "
        "their traits, roles, properties, and stable attributes"
    ),
    "event": (
        "events and state changes — what happened, when it happened, what changed, "
        "in what order. A chronicle of occurrences, not why they occurred"
    ),
    "cause": (
        "causal links between events — why things happened, what led to what, "
        "what enabled or prevented what. Only the explanatory links, not the events themselves"
    ),
    "monolith4": (
        "all aspects: entity identities, relationships, events, and causal patterns"
    ),
}

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_RECORD_SYSTEM = """\
You are the {facet} memory agent. You maintain a notebook of {facet_desc}.
When given a new episode, update your notebook to incorporate any relevant information.
Preserve important details and evolve your understanding over time."""

_RECORD_USER = """\
CURRENT NOTEBOOK:
{notebook}

NEW EPISODE [{episode_id}] ({timestamp}):
{text}

Update the notebook to incorporate this episode. Return the complete updated notebook."""

_CONSULT_SYSTEM = """\
You are the {facet} memory specialist. You have a notebook of {facet_desc}.
Answer the question using ONLY information from your notebook. Cite specific details."""

_CONSULT_USER = """\
NOTEBOOK:
{notebook}

QUESTION: {question}

Answer based on your notebook. Be specific and cite details."""


def _build_synthesis_system(facets: tuple[str, ...]) -> str:
    facet_list = ", ".join(facets)
    return (
        f"You are a synthesis agent. You receive specialist responses from "
        f"{len(facets)} memory facets ({facet_list}) and must combine them "
        f"into a single coherent answer."
    )


def _build_synthesis_user(
    question: str,
    facets: tuple[str, ...],
    facet_responses: dict[str, str],
) -> str:
    parts = [f"QUESTION: {question}"]
    for facet in facets:
        response = facet_responses.get(facet, "(no response)")
        parts.append(f"\n{facet.upper()} SPECIALIST:\n{response}")
    parts.append(
        "\nSynthesize these perspectives into a single, coherent answer. "
        "Resolve any contradictions. Cite the most relevant details from each facet."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _strip_provider_prefix(model: str) -> str:
    if "/" in model and model.startswith(("together/", "openai/")):
        return model.split("/", 1)[1]
    return model


def _complete(
    client: _OpenAI,  # type: ignore[valid-type]
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1500,
) -> str:
    """Single chat completion call."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _TriadBase(MemoryAdapter):
    """Shared logic for all triad variants."""

    _notebook_keys: tuple[str, ...] = ()
    _adapter_label: str = ""

    def __init__(self) -> None:
        self._episodes: list[dict] = []
        self._notebooks: dict[str, str] = {}
        self._oai: _OpenAI | None = None  # type: ignore[valid-type]
        self._model: str = ""

    def reset(self, scope_id: str) -> None:
        self._episodes = []
        self._notebooks = {k: "(empty)" for k in self._notebook_keys}
        self._oai = None
        self._model = ""

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        self._episodes.append({
            "episode_id": episode_id,
            "scope_id": scope_id,
            "timestamp": timestamp,
            "text": text,
            "meta": meta or {},
        })

    def _init_client(self) -> None:
        if _OpenAI is None:
            raise RuntimeError("openai package required for triad adapters")
        api_key = (
            os.environ.get("SYNIX_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        base_url = (
            os.environ.get("SYNIX_LLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model_raw = os.environ.get("SYNIX_LLM_MODEL", "gpt-4o-mini")
        self._model = _strip_provider_prefix(model_raw)

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._oai = _OpenAI(**kwargs)

    def retrieve(self, ref_id: str) -> Document | None:
        if ref_id.startswith("notebook-"):
            facet = ref_id[len("notebook-"):]
            if facet in self._notebooks:
                return Document(ref_id=ref_id, text=self._notebooks[facet])
            return None
        for ep in self._episodes:
            if ep["episode_id"] == ref_id:
                return Document(
                    ref_id=ref_id,
                    text=ep["text"],
                    metadata=ep.get("meta", {}),
                )
        return None

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["synthesis"],
            max_results_per_search=1,
        )

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        refs = []
        for key, content in self._notebooks.items():
            if content and content != "(empty)":
                refs.append((f"notebook-{key}", content))
        return refs

    def _fallback_search(self, limit: int | None) -> list[SearchResult]:
        cap = limit or 10
        return [
            SearchResult(
                ref_id=ep["episode_id"],
                text=ep["text"][:500],
                score=0.5,
                metadata=ep.get("meta", {}),
            )
            for ep in self._episodes[:cap]
        ]


# ---------------------------------------------------------------------------
# Monolith (single agent, single notebook)
# ---------------------------------------------------------------------------

class _MonolithBase(_TriadBase):
    """Single agent maintaining one unified notebook. Subclassed per facet set."""

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._episodes:
            return
        try:
            self._init_client()
        except RuntimeError as e:
            logger.error("%s init failed: %s", self._adapter_label, e)
            return

        nb_key = self._notebook_keys[0]
        for ep in self._episodes:
            try:
                self._notebooks[nb_key] = _complete(
                    self._oai,
                    self._model,
                    system=_RECORD_SYSTEM.format(
                        facet=nb_key,
                        facet_desc=_FACET_DESCRIPTIONS[nb_key],
                    ),
                    user=_RECORD_USER.format(
                        notebook=self._notebooks[nb_key],
                        episode_id=ep["episode_id"],
                        timestamp=ep["timestamp"],
                        text=ep["text"],
                    ),
                )
            except Exception as e:
                logger.error(
                    "%s record failed for episode %s: %s",
                    self._adapter_label, ep["episode_id"], e,
                )

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        nb_key = self._notebook_keys[0]
        if not self._oai or self._notebooks.get(nb_key) == "(empty)":
            return self._fallback_search(limit)

        try:
            answer = _complete(
                self._oai,
                self._model,
                system=_CONSULT_SYSTEM.format(
                    facet=nb_key,
                    facet_desc=_FACET_DESCRIPTIONS[nb_key],
                ),
                user=_CONSULT_USER.format(
                    notebook=self._notebooks[nb_key],
                    question=query,
                ),
            )
            return [SearchResult(
                ref_id=f"{self._adapter_label}-answer",
                text=answer[:500],
                score=1.0,
                metadata={
                    "type": self._adapter_label.replace("-", "_"),
                    "full_answer": answer,
                },
            )]
        except Exception as e:
            logger.error("%s consult failed: %s", self._adapter_label, e)
            return self._fallback_search(limit)


@register_adapter("triad-monolith")
class TriadMonolithAdapter(_MonolithBase):
    """Single agent maintaining one unified notebook (3-facet description)."""

    _notebook_keys = ("monolith",)
    _adapter_label = "triad-monolith"


@register_adapter("triad4-monolith")
class Triad4MonolithAdapter(_MonolithBase):
    """Single agent maintaining one unified notebook (4-facet description)."""

    _notebook_keys = ("monolith4",)
    _adapter_label = "triad4-monolith"


# ---------------------------------------------------------------------------
# Panel (parallel specialists)
# ---------------------------------------------------------------------------

class _PanelBase(_TriadBase):
    """Parallel specialist agents. Subclassed per facet set."""

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._episodes:
            return
        try:
            self._init_client()
        except RuntimeError as e:
            logger.error("%s init failed: %s", self._adapter_label, e)
            return

        for ep in self._episodes:
            self._record_episode_parallel(ep)

    def _record_episode_parallel(self, ep: dict) -> None:
        """Update all facet notebooks in parallel for a single episode."""
        facets = self._notebook_keys

        def update_facet(facet: str) -> tuple[str, str]:
            result = _complete(
                self._oai,
                self._model,
                system=_RECORD_SYSTEM.format(
                    facet=facet,
                    facet_desc=_FACET_DESCRIPTIONS[facet],
                ),
                user=_RECORD_USER.format(
                    notebook=self._notebooks[facet],
                    episode_id=ep["episode_id"],
                    timestamp=ep["timestamp"],
                    text=ep["text"],
                ),
            )
            return facet, result

        with ThreadPoolExecutor(max_workers=len(facets)) as pool:
            futures = {pool.submit(update_facet, f): f for f in facets}
            for fut in as_completed(futures):
                facet = futures[fut]
                try:
                    _, updated = fut.result()
                    self._notebooks[facet] = updated
                except Exception as e:
                    logger.error(
                        "%s record failed for facet=%s episode=%s: %s",
                        self._adapter_label, facet, ep["episode_id"], e,
                    )

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        facets = self._notebook_keys
        if not self._oai or all(
            self._notebooks.get(f) == "(empty)" for f in facets
        ):
            return self._fallback_search(limit)

        facet_responses: dict[str, str] = {}

        def consult_facet(facet: str) -> tuple[str, str]:
            result = _complete(
                self._oai,
                self._model,
                system=_CONSULT_SYSTEM.format(
                    facet=facet,
                    facet_desc=_FACET_DESCRIPTIONS[facet],
                ),
                user=_CONSULT_USER.format(
                    notebook=self._notebooks[facet],
                    question=query,
                ),
            )
            return facet, result

        try:
            with ThreadPoolExecutor(max_workers=len(facets)) as pool:
                futures = {pool.submit(consult_facet, f): f for f in facets}
                for fut in as_completed(futures):
                    facet = futures[fut]
                    try:
                        _, response = fut.result()
                        facet_responses[facet] = response
                    except Exception as e:
                        logger.error(
                            "%s consult failed for facet=%s: %s",
                            self._adapter_label, facet, e,
                        )
                        facet_responses[facet] = "(no response)"

            answer = _complete(
                self._oai,
                self._model,
                system=_build_synthesis_system(facets),
                user=_build_synthesis_user(query, facets, facet_responses),
                max_tokens=2000,
            )
            return [SearchResult(
                ref_id=f"{self._adapter_label}-answer",
                text=answer[:500],
                score=1.0,
                metadata={
                    "type": self._adapter_label.replace("-", "_"),
                    "full_answer": answer,
                },
            )]
        except Exception as e:
            logger.error("%s search failed: %s", self._adapter_label, e)
            return self._fallback_search(limit)


@register_adapter("triad-panel")
class TriadPanelAdapter(_PanelBase):
    """Three parallel specialist agents (identity, relation, causation)."""

    _notebook_keys = FACETS
    _adapter_label = "triad-panel"


@register_adapter("triad4-panel")
class Triad4PanelAdapter(_PanelBase):
    """Four parallel specialist agents (entity, relation, event, cause)."""

    _notebook_keys = FACETS_4
    _adapter_label = "triad4-panel"


# ---------------------------------------------------------------------------
# Conversation (sequential specialists)
# ---------------------------------------------------------------------------

class _ConversationBase(_TriadBase):
    """Sequential specialist agents. Subclassed per facet set."""

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._episodes:
            return
        try:
            self._init_client()
        except RuntimeError as e:
            logger.error("%s init failed: %s", self._adapter_label, e)
            return

        # Episodes sequential, facets parallel (same as panel for recording).
        for ep in self._episodes:
            self._record_episode_parallel(ep)

    def _record_episode_parallel(self, ep: dict) -> None:
        """Update all facet notebooks in parallel for a single episode."""
        facets = self._notebook_keys

        def update_facet(facet: str) -> tuple[str, str]:
            result = _complete(
                self._oai,
                self._model,
                system=_RECORD_SYSTEM.format(
                    facet=facet,
                    facet_desc=_FACET_DESCRIPTIONS[facet],
                ),
                user=_RECORD_USER.format(
                    notebook=self._notebooks[facet],
                    episode_id=ep["episode_id"],
                    timestamp=ep["timestamp"],
                    text=ep["text"],
                ),
            )
            return facet, result

        with ThreadPoolExecutor(max_workers=len(facets)) as pool:
            futures = {pool.submit(update_facet, f): f for f in facets}
            for fut in as_completed(futures):
                facet = futures[fut]
                try:
                    _, updated = fut.result()
                    self._notebooks[facet] = updated
                except Exception as e:
                    logger.error(
                        "%s record failed for facet=%s episode=%s: %s",
                        self._adapter_label, facet, ep["episode_id"], e,
                    )

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        facets = self._notebook_keys
        if not self._oai or all(
            self._notebooks.get(f) == "(empty)" for f in facets
        ):
            return self._fallback_search(limit)

        # Sequential consultation in facet order.
        # For 3-facet: identity → relation → causation
        # For 4-facet: entity → relation → event → cause
        try:
            prior_context = ""
            facet_responses: dict[str, str] = {}

            for facet in facets:
                augmented_question = query
                if prior_context:
                    augmented_question = (
                        f"{query}\n\nPrior specialist context:\n{prior_context}"
                    )

                response = _complete(
                    self._oai,
                    self._model,
                    system=_CONSULT_SYSTEM.format(
                        facet=facet,
                        facet_desc=_FACET_DESCRIPTIONS[facet],
                    ),
                    user=_CONSULT_USER.format(
                        notebook=self._notebooks[facet],
                        question=augmented_question,
                    ),
                )
                facet_responses[facet] = response
                prior_context = f"[{facet}]: {response}"

            answer = _complete(
                self._oai,
                self._model,
                system=_build_synthesis_system(facets),
                user=_build_synthesis_user(query, facets, facet_responses),
                max_tokens=2000,
            )
            return [SearchResult(
                ref_id=f"{self._adapter_label}-answer",
                text=answer[:500],
                score=1.0,
                metadata={
                    "type": self._adapter_label.replace("-", "_"),
                    "full_answer": answer,
                },
            )]
        except Exception as e:
            logger.error("%s search failed: %s", self._adapter_label, e)
            return self._fallback_search(limit)


@register_adapter("triad-conversation")
class TriadConversationAdapter(_ConversationBase):
    """Three sequential specialists — identity → relation → causation."""

    _notebook_keys = FACETS
    _adapter_label = "triad-conversation"


@register_adapter("triad4-conversation")
class Triad4ConversationAdapter(_ConversationBase):
    """Four sequential specialists — entity → relation → event → cause."""

    _notebook_keys = FACETS_4
    _adapter_label = "triad4-conversation"
