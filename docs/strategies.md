# SWE-bench Context Management Strategies

synix-bench includes 10 context management strategies for SWE-bench coding tasks. Each strategy controls how the agent's conversation history is managed as it works through a multi-step coding task.

## Strategy Overview

All strategies share the same interface:

```python
class ContextStrategy(Protocol):
    name: str

    def run(self, client: OpenAI, model: str, task: str,
            executor: ToolExecutor, max_steps: int = 30, **kwargs) -> dict:
        """Execute the agent loop with this context management approach."""
```

The agent loop follows the same pattern in every strategy:

1. Build messages (system + history, managed per strategy)
2. Call LLM with tool definitions
3. Parse tool calls from response
4. Execute tools via `ToolExecutor` (runs inside a container)
5. Append results to history
6. Apply context management (strategy-specific)
7. Repeat until done or budget exhausted

## Strategies

### `naive`

**No context management.** Full message history is sent to the LLM on every turn.

- Baseline for comparison
- Works well with large context windows (128K+)
- Token usage grows quadratically with conversation length
- Context overflow errors with smaller models

```bash
synix-bench run --suite swebench --strategy naive
```

### `window`

**Sliding window.** Keep the last N messages, drop the oldest.

- Default window: 20 messages
- Simple and predictable
- Loses early context (task description, initial exploration)
- Good balance of cost and quality for medium-length tasks

```bash
synix-bench run --suite swebench --strategy window
```

### `truncation`

**Token budget truncation.** Drop oldest messages until the total is under a token budget.

- Default budget: 8K tokens
- More precise than window (accounts for varying message sizes)
- Same drawback: loses early context

```bash
synix-bench run --suite swebench --strategy truncation
```

### `summary`

**Periodic LLM summarization.** Every K steps, ask the LLM to summarize the conversation so far, then replace history with the summary.

- Default interval: every 5 steps
- Preserves semantic content across the full history
- Costs extra LLM calls for summarization
- Summary quality depends on the model

```bash
synix-bench run --suite swebench --strategy summary
```

### `masking`

**Output masking.** Replace old tool outputs with short placeholders, keeping the tool call structure visible.

- Preserves the "shape" of the conversation (what tools were called and when)
- Dramatically reduces token count for verbose tool outputs
- Agent can see what it did but not the full output details

```bash
synix-bench run --suite swebench --strategy masking
```

### `rag`

**Retrieval-augmented generation.** Maintains a memory bank of past observations. On each turn, retrieves the most relevant entries via BM25 search and combines with a recent-message window.

- Keeps a small recent window (last 5 messages)
- Retrieves up to 5 relevant past observations per turn
- BM25 keyword matching (no embedding model needed)
- Good for tasks where the agent needs to recall specific past findings

```bash
synix-bench run --suite swebench --strategy rag
```

### `incremental_summary`

**Rolling compression.** After each step, incrementally update a running summary. The summary grows monotonically but much slower than raw history.

- One LLM summarization call per step
- Summary carries forward the most important information
- Higher cost than periodic summary, but no information cliff edges
- Best for long tasks where gradual context loss is unacceptable

```bash
synix-bench run --suite swebench --strategy incremental_summary
```

### `structured_summary`

**JSON-schema summarization.** Periodically summarize into a structured JSON format with fields for current state, findings, plan, and blockers.

- Structured format ensures consistent information retention
- Easier for the agent to parse and use compared to free-text summaries
- Enforced via JSON schema in the summarization prompt

```bash
synix-bench run --suite swebench --strategy structured_summary
```

### `hierarchical`

**Three-tier memory.** Maintains hot (recent), warm (masked), and cold (summarized) tiers.

- Hot tier: last 5 messages (full content)
- Warm tier: next 15 messages (tool outputs masked)
- Cold tier: everything older (LLM-summarized)
- Combines the benefits of masking and summarization
- Agent has full recent context + structure of recent past + summary of distant past

```bash
synix-bench run --suite swebench --strategy hierarchical
```

### `stack_heap`

**Agent-controlled structured memory.** The agent explicitly manages its own context through stack and heap operations.

- **Stack**: push/pop frames with objectives, context, and return specs
- **Heap**: alloc/write/free named memory blocks with size tracking
- Agent follows a priority-ordered action loop (free > write > pop > push > alloc > work)
- Requires a specialized system prompt (see `prompts/stack-heap-v7.txt`)
- Requires a layout config (see `configs/layouts/`)

```bash
synix-bench run --suite swebench --strategy stack_heap \
  --layout configs/layouts/order-conv-mask.json
```

Layout configs control section ordering, masking behavior, and register sizes. Available layouts are in `configs/layouts/`.

## Comparing Strategies

Run multiple strategies against the same instances:

```bash
# Single instance comparison
for strategy in naive window truncation summary masking; do
  synix-bench run --suite swebench --strategy $strategy \
    --instance-id django__django-16139 \
    --model gpt-4o --api-key-env OPENAI_API_KEY \
    --trials 5
done
```

Or use a sweep config:

```json
{
  "configs": [
    {"suite": "swebench", "strategy": "naive", "llm": {"model": "gpt-4o"}, "instance_id": "django__django-16139", "trials": 5},
    {"suite": "swebench", "strategy": "window", "llm": {"model": "gpt-4o"}, "instance_id": "django__django-16139", "trials": 5},
    {"suite": "swebench", "strategy": "masking", "llm": {"model": "gpt-4o"}, "instance_id": "django__django-16139", "trials": 5}
  ]
}
```

## Adding a New Strategy

See [extending.md](extending.md) for how to implement and register a new strategy.
