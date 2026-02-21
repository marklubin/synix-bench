# LENS Memory Adapters

The LENS suite evaluates memory system backends through a standardized adapter interface. Each adapter implements the `MemoryAdapter` ABC and is registered via the `@register_adapter` decorator.

## Adapter Interface

```python
class MemoryAdapter(ABC):
    """Abstract base class for memory system adapters."""

    # --- Data loading (called by runner) ---

    def reset(self, scope_id: str) -> None:
        """Clear all state for a scope."""

    def ingest(self, episode_id, scope_id, timestamp, text, meta=None) -> None:
        """Ingest a single episode. Must complete within 200ms, no LLM calls."""

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Optional: build indices or consolidate before questions."""

    # --- Tools (exposed to agent) ---

    def search(self, query, filters=None, limit=None) -> list[SearchResult]:
        """Search memory for relevant information."""

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve a full document by reference ID."""

    def get_capabilities(self) -> CapabilityManifest:
        """Declare supported search modes, filters, and tools."""

    # --- Optional ---

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        """Return adapter-generated documents (summaries, consolidations)."""

    def get_cache_state(self) -> dict | None:
        """Return serializable state for caching."""

    def restore_cache_state(self, state: dict) -> bool:
        """Restore from cached state."""

    def call_extended_tool(self, tool_name, arguments) -> object:
        """Handle adapter-defined extended tools."""
```

## Data Types

### SearchResult

```python
@dataclass(frozen=True)
class SearchResult:
    ref_id: str          # Reference ID (used for citation tracking)
    text: str            # Result text content
    score: float = 0.0   # Relevance score
    metadata: dict = {}   # Optional metadata
```

### Document

```python
@dataclass(frozen=True)
class Document:
    ref_id: str          # Reference ID
    text: str            # Full document text
    metadata: dict = {}   # Optional metadata
```

### CapabilityManifest

The manifest tells the agent harness what the adapter supports, so tool definitions are generated dynamically:

```python
@dataclass
class CapabilityManifest:
    search_modes: list[str] = ["semantic"]
    filter_fields: list[FilterField] = []
    max_results_per_search: int = 10
    supports_date_range: bool = False
    extra_tools: list[ExtraTool] = []
```

## Built-in Adapters

### Null Adapter (`null`)

No-op adapter for baseline measurements. All searches return empty results.

### SQLite Family

Local SQLite-based adapters with progressively more sophisticated search:

| Adapter | Search Method |
|---|---|
| `sqlite` | Exact substring match |
| `sqlite-fts` | SQLite FTS5 full-text search |
| `sqlite-chunked` | Chunked document storage + substring search |
| `sqlite-embedding` | Local embedding cosine similarity |
| `sqlite-embedding-openai` | OpenAI embedding cosine similarity |
| `sqlite-hybrid` | FTS5 + local embedding fusion |
| `sqlite-hybrid-openai` | FTS5 + OpenAI embedding fusion |
| `sqlite-chunked-hybrid` | Chunked storage + hybrid search |

All SQLite adapters are self-contained (no external services besides optional OpenAI for embeddings).

### External Service Adapters

| Adapter | Backend | Requires |
|---|---|---|
| `mem0-raw` | Mem0 raw storage | `mem0ai` package |
| `mem0-extract` | Mem0 with extraction | `mem0ai` package |
| `letta` | Letta agent memory | `letta-client` package + Letta server |
| `letta-sleepy` | Letta with sleepy retention | `letta-client` package + Letta server |
| `graphiti` | Graphiti knowledge graph | `graphiti-core` package + Neo4j |
| `cognee` | Cognee cognitive memory | `cognee` package |
| `compaction` | LLM-driven compaction | OpenAI API key |
| `hindsight` | Hindsight consolidation | OpenAI API key |

Install external adapters:

```bash
uv sync --extra adapters  # mem0ai, letta-client
```

## LENS Evaluation Flow

1. **Reset**: `adapter.reset(scope_id)` clears all state
2. **Ingest**: Episodes are streamed in chronological order via `adapter.ingest()`
3. **Prepare**: `adapter.prepare(scope_id, checkpoint)` allows index building
4. **Register synthetic refs**: Adapter-generated documents are added to the EpisodeVault
5. **Answer questions**: AgentHarness calls `search()`, `retrieve()`, and extended tools
6. **Verify**: Check that cited references exist in the EpisodeVault

## Writing a New Adapter

See [extending.md](extending.md) for a step-by-step guide.

### Quick Example

```python
# src/synix/suites/lens/adapters/my_adapter.py

from synix.suites.lens.adapters.base import (
    CapabilityManifest, Document, MemoryAdapter, SearchResult,
)
from synix.suites.lens.adapters.registry import register_adapter


@register_adapter("my-adapter")
class MyAdapter(MemoryAdapter):
    def __init__(self):
        self._docs: dict[str, str] = {}

    def reset(self, scope_id: str) -> None:
        self._docs.clear()

    def ingest(self, episode_id, scope_id, timestamp, text, meta=None) -> None:
        self._docs[episode_id] = text

    def search(self, query, filters=None, limit=None) -> list[SearchResult]:
        results = []
        for ref_id, text in self._docs.items():
            if query.lower() in text.lower():
                results.append(SearchResult(ref_id=ref_id, text=text[:200]))
        return results[:limit or 10]

    def retrieve(self, ref_id: str) -> Document | None:
        text = self._docs.get(ref_id)
        return Document(ref_id=ref_id, text=text) if text else None

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(search_modes=["keyword"])
```

Then register it in the lazy loader at `src/synix/suites/lens/adapters/registry.py`.

## Conformance Tests

Run the adapter conformance suite to verify your adapter meets all interface requirements:

```bash
uv run pytest tests/conformance/test_adapter_conformance.py -v
```

The conformance tests check:
- `reset()` clears state
- `ingest()` stores episodes
- `search()` returns results after ingestion
- `retrieve()` returns documents by ref_id
- `get_capabilities()` returns a valid manifest
