"""Unit tests for Triad Memory Protocol adapters (3-facet and 4-facet).

Tests all six triad adapters covering:
- Registration in the adapter registry
- Lifecycle: reset → ingest → prepare → search → retrieve
- Buffering without LLM calls
- Notebook state management
- Fallback behavior when no LLM is configured
- Mocked LLM paths for prepare() and search()
- 4-facet notebook decomposition
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from synix.suites.lens.adapters.base import (
    CapabilityManifest,
    Document,
    SearchResult,
)
from synix.suites.lens.adapters.registry import get_adapter, list_adapters
from synix.suites.lens.adapters.triad import (
    FACETS,
    FACETS_4,
    TriadMonolithAdapter,
    TriadPanelAdapter,
    TriadConversationAdapter,
    Triad4MonolithAdapter,
    Triad4PanelAdapter,
    Triad4ConversationAdapter,
    _TriadBase,
    _complete,
    _strip_provider_prefix,
    _build_synthesis_system,
    _build_synthesis_user,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_EPISODES = [
    ("ep-1", "scope-a", "2024-01-01T09:00:00", "Alice is a software engineer at Acme Corp."),
    ("ep-2", "scope-a", "2024-01-02T10:00:00", "Bob is Alice's manager. He mentors her."),
    ("ep-3", "scope-a", "2024-01-03T11:00:00", "Alice's debugging caused the server outage to be fixed in 2 hours."),
]


def _ingest_samples(adapter: _TriadBase) -> None:
    adapter.reset("scope-a")
    for eid, sid, ts, text in SAMPLE_EPISODES:
        adapter.ingest(eid, sid, ts, text)


def _make_mock_completion(responses: list[str] | None = None) -> MagicMock:
    """Create a mock OpenAI client whose chat.completions.create returns canned responses."""
    idx = {"i": 0}
    defaults = responses or ["mock response"]

    def side_effect(**kwargs):
        text = defaults[idx["i"] % len(defaults)]
        idx["i"] += 1
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=side_effect)
    return client


def _setup_mock(adapter: _TriadBase, responses: list[str]) -> MagicMock:
    """Inject a mock client and patch _init_client so prepare() doesn't overwrite it."""
    mock_client = _make_mock_completion(responses)
    adapter._oai = mock_client
    adapter._model = "test-model"
    adapter._init_client = lambda: None  # type: ignore[assignment]
    return mock_client


ALL_ADAPTER_CLASSES = [
    TriadMonolithAdapter,
    TriadPanelAdapter,
    TriadConversationAdapter,
    Triad4MonolithAdapter,
    Triad4PanelAdapter,
    Triad4ConversationAdapter,
]

ALL_ADAPTER_NAMES = [
    "triad-monolith", "triad-panel", "triad-conversation",
    "triad4-monolith", "triad4-panel", "triad4-conversation",
]


# ===========================================================================
# Registry tests
# ===========================================================================


class TestTriadRegistry:
    def test_all_six_registered(self):
        adapters = list_adapters()
        for name in ALL_ADAPTER_NAMES:
            assert name in adapters, f"{name} not registered"

    @pytest.mark.parametrize("name,cls", [
        ("triad-monolith", TriadMonolithAdapter),
        ("triad-panel", TriadPanelAdapter),
        ("triad-conversation", TriadConversationAdapter),
        ("triad4-monolith", Triad4MonolithAdapter),
        ("triad4-panel", Triad4PanelAdapter),
        ("triad4-conversation", Triad4ConversationAdapter),
    ])
    def test_get_adapter_returns_correct_class(self, name: str, cls: type):
        assert get_adapter(name) is cls


# ===========================================================================
# _strip_provider_prefix
# ===========================================================================


class TestStripProviderPrefix:
    def test_strips_openai_prefix(self):
        assert _strip_provider_prefix("openai/gpt-4o") == "gpt-4o"

    def test_strips_together_prefix(self):
        assert _strip_provider_prefix("together/meta-llama/Llama-3-70b") == "meta-llama/Llama-3-70b"

    def test_leaves_bare_model_alone(self):
        assert _strip_provider_prefix("gpt-4o-mini") == "gpt-4o-mini"

    def test_leaves_unknown_prefix_alone(self):
        assert _strip_provider_prefix("anthropic/claude-3") == "anthropic/claude-3"


# ===========================================================================
# Synthesis prompt builders
# ===========================================================================


class TestSynthesisPrompts:
    def test_3_facet_system(self):
        s = _build_synthesis_system(FACETS)
        assert "3 memory facets" in s
        assert "identity, relation, causation" in s

    def test_4_facet_system(self):
        s = _build_synthesis_system(FACETS_4)
        assert "4 memory facets" in s
        assert "entity, relation, event, cause" in s

    def test_3_facet_user(self):
        responses = {"identity": "id resp", "relation": "rel resp", "causation": "cause resp"}
        u = _build_synthesis_user("What?", FACETS, responses)
        assert "IDENTITY SPECIALIST:" in u
        assert "RELATION SPECIALIST:" in u
        assert "CAUSATION SPECIALIST:" in u
        assert "id resp" in u
        assert "What?" in u

    def test_4_facet_user(self):
        responses = {
            "entity": "ent resp", "relation": "rel resp",
            "event": "evt resp", "cause": "cau resp",
        }
        u = _build_synthesis_user("Why?", FACETS_4, responses)
        assert "ENTITY SPECIALIST:" in u
        assert "RELATION SPECIALIST:" in u
        assert "EVENT SPECIALIST:" in u
        assert "CAUSE SPECIALIST:" in u

    def test_missing_response_defaults(self):
        u = _build_synthesis_user("Q?", FACETS, {})
        assert "(no response)" in u


# ===========================================================================
# Shared base behavior (tested via each concrete subclass)
# ===========================================================================


@pytest.fixture(params=ALL_ADAPTER_CLASSES, ids=[
    "monolith", "panel", "conversation",
    "monolith4", "panel4", "conversation4",
])
def adapter(request) -> _TriadBase:
    return request.param()


class TestTriadBaseLifecycle:
    """Tests that apply to all six triad adapters."""

    def test_reset_clears_state(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert len(adapter._episodes) == 3
        adapter.reset("scope-b")
        assert adapter._episodes == []
        for key in adapter._notebook_keys:
            assert adapter._notebooks[key] == "(empty)"

    def test_ingest_buffers_episodes(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert len(adapter._episodes) == 3
        assert adapter._episodes[0]["episode_id"] == "ep-1"
        assert adapter._episodes[2]["text"] == SAMPLE_EPISODES[2][3]

    def test_ingest_stores_meta(self, adapter: _TriadBase):
        adapter.reset("s")
        adapter.ingest("ep-x", "s", "2024-01-01", "text", meta={"tag": "test"})
        assert adapter._episodes[0]["meta"] == {"tag": "test"}

    def test_ingest_defaults_meta_to_empty(self, adapter: _TriadBase):
        adapter.reset("s")
        adapter.ingest("ep-x", "s", "2024-01-01", "text")
        assert adapter._episodes[0]["meta"] == {}

    def test_retrieve_episode_by_id(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        doc = adapter.retrieve("ep-2")
        assert doc is not None
        assert isinstance(doc, Document)
        assert "Bob is Alice's manager" in doc.text

    def test_retrieve_nonexistent_returns_none(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert adapter.retrieve("ep-999") is None

    def test_retrieve_notebook_when_empty(self, adapter: _TriadBase):
        adapter.reset("s")
        key = adapter._notebook_keys[0]
        doc = adapter.retrieve(f"notebook-{key}")
        assert doc is not None
        assert doc.text == "(empty)"

    def test_retrieve_notebook_invalid_facet(self, adapter: _TriadBase):
        adapter.reset("s")
        assert adapter.retrieve("notebook-nonexistent") is None

    def test_get_capabilities(self, adapter: _TriadBase):
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "synthesis" in caps.search_modes
        assert caps.max_results_per_search == 1

    def test_synthetic_refs_empty_before_prepare(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert adapter.get_synthetic_refs() == []

    def test_synthetic_refs_after_notebook_update(self, adapter: _TriadBase):
        adapter.reset("s")
        key = adapter._notebook_keys[0]
        adapter._notebooks[key] = "Updated notebook content"
        refs = adapter.get_synthetic_refs()
        assert len(refs) >= 1
        ref_ids = [r[0] for r in refs]
        assert f"notebook-{key}" in ref_ids

    def test_adapter_label_set(self, adapter: _TriadBase):
        assert adapter._adapter_label != ""
        assert adapter._adapter_label.startswith("triad")


# ===========================================================================
# Notebook key configuration
# ===========================================================================


class TestNotebookKeys:
    def test_monolith_has_single_notebook(self):
        assert TriadMonolithAdapter()._notebook_keys == ("monolith",)

    def test_panel_has_three_notebooks(self):
        a = TriadPanelAdapter()
        assert a._notebook_keys == FACETS
        assert len(a._notebook_keys) == 3

    def test_conversation_has_three_notebooks(self):
        assert TriadConversationAdapter()._notebook_keys == FACETS

    def test_monolith4_has_single_notebook(self):
        assert Triad4MonolithAdapter()._notebook_keys == ("monolith4",)

    def test_panel4_has_four_notebooks(self):
        a = Triad4PanelAdapter()
        assert a._notebook_keys == FACETS_4
        assert len(a._notebook_keys) == 4

    def test_conversation4_has_four_notebooks(self):
        assert Triad4ConversationAdapter()._notebook_keys == FACETS_4

    def test_4facet_ordering(self):
        """Developmental order: entity → relation → event → cause."""
        assert FACETS_4 == ("entity", "relation", "event", "cause")


# ===========================================================================
# Fallback behavior (no LLM configured → returns raw episodes)
# ===========================================================================


class TestFallbackSearch:
    """When no LLM client is available, search() should fall back to raw episodes."""

    @pytest.mark.parametrize("cls", ALL_ADAPTER_CLASSES, ids=[
        "monolith", "panel", "conversation",
        "monolith4", "panel4", "conversation4",
    ])
    def test_fallback_returns_episodes(self, cls: type):
        a = cls()
        _ingest_samples(a)
        results = a.search("anything")
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].ref_id == "ep-1"
        assert results[0].score == 0.5

    def test_fallback_respects_limit(self):
        a = TriadMonolithAdapter()
        _ingest_samples(a)
        results = a.search("anything", limit=2)
        assert len(results) == 2


# ===========================================================================
# Monolith with mocked LLM (both 3-facet and 4-facet)
# ===========================================================================


class TestMonolithWithMockedLLM:
    @pytest.mark.parametrize("cls,nb_key,label", [
        (TriadMonolithAdapter, "monolith", "triad-monolith"),
        (Triad4MonolithAdapter, "monolith4", "triad4-monolith"),
    ], ids=["3-facet", "4-facet"])
    def test_prepare_updates_notebook(self, cls, nb_key, label):
        a = cls()
        _ingest_samples(a)
        mock_client = _setup_mock(a, [
            "Notebook after ep-1",
            "Notebook after ep-2",
            "Notebook after ep-3",
        ])

        a.prepare("scope-a", 1)

        assert mock_client.chat.completions.create.call_count == 3
        assert a._notebooks[nb_key] == "Notebook after ep-3"

    @pytest.mark.parametrize("cls,nb_key,label", [
        (TriadMonolithAdapter, "monolith", "triad-monolith"),
        (Triad4MonolithAdapter, "monolith4", "triad4-monolith"),
    ], ids=["3-facet", "4-facet"])
    def test_search_returns_synthesized_answer(self, cls, nb_key, label):
        a = cls()
        a.reset("s")
        a._oai = _make_mock_completion(["The answer is 42."])
        a._model = "test-model"
        a._notebooks[nb_key] = "Alice is an engineer."

        results = a.search("What is Alice's role?")
        assert len(results) == 1
        assert results[0].ref_id == f"{label}-answer"
        assert results[0].score == 1.0
        assert "42" in results[0].text
        assert results[0].metadata["type"] == label.replace("-", "_")

    def test_search_stores_full_answer_in_metadata(self):
        long_answer = "A" * 1000
        a = TriadMonolithAdapter()
        a.reset("s")
        a._oai = _make_mock_completion([long_answer])
        a._model = "test-model"
        a._notebooks["monolith"] = "content"

        results = a.search("q")
        assert len(results[0].text) == 500  # truncated
        assert results[0].metadata["full_answer"] == long_answer

    def test_prepare_handles_llm_error(self):
        a = TriadMonolithAdapter()
        _ingest_samples(a)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        a._oai = mock_client
        a._model = "test-model"
        a._init_client = lambda: None  # type: ignore[assignment]

        a.prepare("scope-a", 1)
        assert a._notebooks["monolith"] == "(empty)"

    def test_search_handles_llm_error(self):
        a = TriadMonolithAdapter()
        _ingest_samples(a)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        a._oai = mock_client
        a._model = "test-model"
        a._notebooks["monolith"] = "some content"

        results = a.search("q")
        assert len(results) == 3
        assert results[0].score == 0.5


# ===========================================================================
# Panel with mocked LLM (3-facet and 4-facet)
# ===========================================================================


class TestPanelWithMockedLLM:
    @pytest.mark.parametrize("cls,facets,label", [
        (TriadPanelAdapter, FACETS, "triad-panel"),
        (Triad4PanelAdapter, FACETS_4, "triad4-panel"),
    ], ids=["3-facet", "4-facet"])
    def test_prepare_updates_all_notebooks(self, cls, facets, label):
        a = cls()
        a.reset("s")
        a.ingest("ep-1", "s", "2024-01-01", "Alice met Bob.")

        def side_effect(**kwargs):
            system = kwargs["messages"][0]["content"]
            text = "Updated unknown notebook"
            for facet in facets:
                if f"the {facet} memory" in system:
                    text = f"Updated {facet} notebook"
                    break
            msg = MagicMock()
            msg.content = text
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        a._init_client = lambda: None  # type: ignore[assignment]

        a.prepare("s", 1)

        assert mock_client.chat.completions.create.call_count == len(facets)
        for facet in facets:
            assert a._notebooks[facet] == f"Updated {facet} notebook"

    @pytest.mark.parametrize("cls,facets,label", [
        (TriadPanelAdapter, FACETS, "triad-panel"),
        (Triad4PanelAdapter, FACETS_4, "triad4-panel"),
    ], ids=["3-facet", "4-facet"])
    def test_search_consults_then_synthesizes(self, cls, facets, label):
        a = cls()
        a.reset("s")

        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            msg = MagicMock()
            msg.content = f"Response {call_count['n']}"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in facets:
            a._notebooks[f] = f"Content for {f}"

        results = a.search("Tell me about Alice")
        assert len(results) == 1
        assert results[0].ref_id == f"{label}-answer"
        # N consults + 1 synthesize
        assert call_count["n"] == len(facets) + 1


# ===========================================================================
# Conversation with mocked LLM (3-facet and 4-facet)
# ===========================================================================


class TestConversationWithMockedLLM:
    @pytest.mark.parametrize("cls,facets,label", [
        (TriadConversationAdapter, FACETS, "triad-conversation"),
        (Triad4ConversationAdapter, FACETS_4, "triad4-conversation"),
    ], ids=["3-facet", "4-facet"])
    def test_search_is_sequential(self, cls, facets, label):
        """Verify facets are consulted in order, then synthesis."""
        a = cls()
        a.reset("s")

        call_log: list[str] = []

        def side_effect(**kwargs):
            system = kwargs["messages"][0]["content"]
            if "synthesis agent" in system.lower():
                call_log.append("synthesis")
            else:
                matched = False
                for facet in facets:
                    if f"the {facet} memory" in system:
                        call_log.append(facet)
                        matched = True
                        break
                if not matched:
                    call_log.append("unknown")
            msg = MagicMock()
            msg.content = f"Response from {call_log[-1]}"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in facets:
            a._notebooks[f] = f"Content for {f}"

        results = a.search("What happened?")
        assert len(results) == 1
        assert results[0].ref_id == f"{label}-answer"

        expected = list(facets) + ["synthesis"]
        assert call_log == expected

    def test_prior_context_is_passed_3facet(self):
        """Verify that each specialist receives prior specialist's output (3-facet)."""
        a = TriadConversationAdapter()
        a.reset("s")

        captured_users: list[str] = []

        def side_effect(**kwargs):
            user_msg = kwargs["messages"][1]["content"]
            captured_users.append(user_msg)
            msg = MagicMock()
            msg.content = "Answer"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS:
            a._notebooks[f] = f"Content for {f}"

        a.search("Who is Alice?")

        # First call (identity) — no prior context
        assert "Prior specialist context" not in captured_users[0]
        # Second call (relation) — has identity's output
        assert "Prior specialist context" in captured_users[1]
        assert "[identity]" in captured_users[1]
        # Third call (causation) — has relation's output
        assert "Prior specialist context" in captured_users[2]
        assert "[relation]" in captured_users[2]

    def test_prior_context_is_passed_4facet(self):
        """Verify sequential chaining for 4-facet: entity → relation → event → cause."""
        a = Triad4ConversationAdapter()
        a.reset("s")

        captured_users: list[str] = []

        def side_effect(**kwargs):
            user_msg = kwargs["messages"][1]["content"]
            captured_users.append(user_msg)
            msg = MagicMock()
            msg.content = "Answer"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        a.search("Why?")

        # entity (1st) — no prior
        assert "Prior specialist context" not in captured_users[0]
        # relation (2nd) — has entity
        assert "[entity]" in captured_users[1]
        # event (3rd) — has relation
        assert "[relation]" in captured_users[2]
        # cause (4th) — has event
        assert "[event]" in captured_users[3]


# ===========================================================================
# _complete() helper
# ===========================================================================


class TestCompleteHelper:
    def test_calls_openai_correctly(self):
        mock_client = _make_mock_completion(["hello world"])
        result = _complete(mock_client, "gpt-4o", "system msg", "user msg", max_tokens=100)
        assert result == "hello world"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "system msg"},
                {"role": "user", "content": "user msg"},
            ],
            max_tokens=100,
            temperature=0.0,
        )

    def test_returns_empty_on_none_content(self):
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        result = _complete(mock_client, "m", "s", "u")
        assert result == ""


# ===========================================================================
# _init_client env var handling
# ===========================================================================


class TestInitClient:
    @patch.dict("os.environ", {
        "SYNIX_LLM_API_KEY": "test-key",
        "SYNIX_LLM_API_BASE": "http://localhost:8080/v1",
        "SYNIX_LLM_MODEL": "together/meta-llama/Llama-3-70b",
    })
    @patch("synix.suites.lens.adapters.triad._OpenAI")
    def test_uses_env_vars(self, mock_openai_cls):
        a = TriadMonolithAdapter()
        a.reset("s")
        a._init_client()
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="http://localhost:8080/v1",
        )
        assert a._model == "meta-llama/Llama-3-70b"

    @patch.dict("os.environ", {}, clear=True)
    @patch("synix.suites.lens.adapters.triad._OpenAI")
    def test_defaults(self, mock_openai_cls):
        a = TriadMonolithAdapter()
        a.reset("s")
        a._init_client()
        mock_openai_cls.assert_called_once_with(api_key="dummy")
        assert a._model == "gpt-4o-mini"

    @patch("synix.suites.lens.adapters.triad._OpenAI", None)
    def test_raises_without_openai_package(self):
        a = TriadMonolithAdapter()
        a.reset("s")
        with pytest.raises(RuntimeError, match="openai package required"):
            a._init_client()
