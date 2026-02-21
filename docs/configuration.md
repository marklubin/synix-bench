# Configuration Reference

synix-bench runs can be configured via CLI flags or a JSON config file. CLI flags take precedence over config file values.

## Config File Format

```json
{
  "suite": "lens",
  "strategy": "sqlite-hybrid",
  "dataset": "datasets/smoke/smoke_dataset.json",
  "output_dir": "results",
  "seed": 42,
  "trials": 1,

  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_base": "https://api.openai.com/v1",
    "api_key": null,
    "temperature": 0.0,
    "max_tokens": 4096,
    "seed": 42,
    "extra_body": null
  },

  "agent_budget": {
    "preset": "standard",
    "max_turns": 10,
    "max_tool_calls": 20,
    "max_payload_bytes": 65536,
    "max_latency_per_call_ms": 5000,
    "max_agent_tokens": 32768,
    "max_cumulative_result_tokens": 0
  },

  "swebench": {
    "max_steps": 30,
    "timeout": 1800,
    "workers": 6,
    "layout_file": null,
    "no_think_prefill": false
  },

  "infra": {
    "provider": "local",
    "container_cmd": "podman",
    "docker_host": null,
    "ghcr_prefix": "ghcr.io/marklubin/swebench"
  },

  "sample": null,
  "instance_id": null,
  "checkpoints": [10, 20, 40, 80],
  "parallel_questions": 1,
  "cache_dir": null
}
```

## Top-Level Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `suite` | string | `"lens"` | Benchmark suite: `"swebench"` or `"lens"` |
| `strategy` | string | `"null"` | Strategy (SWE-bench) or adapter (LENS) name |
| `dataset` | string | `""` | Path to dataset file. Empty = use bundled default |
| `output_dir` | string | `"results"` | Directory for result JSON files |
| `seed` | int | `42` | Random seed for reproducibility |
| `trials` | int | `1` | Number of trials per task |
| `sample` | int\|null | `null` | Number of instances to sample from dataset |
| `instance_id` | string\|null | `null` | Run a specific instance by ID |
| `checkpoints` | list[int] | `[10, 20, 40, 80]` | LENS checkpoint episode counts |
| `parallel_questions` | int | `1` | LENS concurrent question threads |
| `cache_dir` | string\|null | `null` | LENS adapter cache directory |

## LLM Configuration (`llm`)

| Field | Type | Default | Env Override | Description |
|---|---|---|---|---|
| `provider` | string | `"mock"` | `SYNIX_LLM_PROVIDER` | `"mock"` or `"openai"` |
| `model` | string | `"gpt-4o-mini"` | `SYNIX_LLM_MODEL` | Model name/ID |
| `api_base` | string\|null | `null` | `SYNIX_LLM_API_BASE` | API endpoint URL |
| `api_key` | string\|null | `null` | `SYNIX_LLM_API_KEY` | API key |
| `temperature` | float | `0.0` | `SYNIX_LLM_TEMPERATURE` | Sampling temperature |
| `max_tokens` | int | `4096` | - | Max output tokens |
| `seed` | int | `42` | `SYNIX_LLM_SEED` | Deterministic seed |
| `extra_body` | dict\|null | `null` | - | Extra request body fields |

The `api_key` field is resolved from environment variables. Use `--api-key-env` in the CLI to specify which env var to read.

For Qwen3 models on vLLM, disable think mode:

```json
{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
```

Or via CLI: `--no-think-prefill`.

## Agent Budget Configuration (`agent_budget`)

Controls resource limits for the LENS agent harness.

| Field | Type | Default | Enforcement |
|---|---|---|---|
| `preset` | string | `"standard"` | Selects default values |
| `max_turns` | int | `10` | Hard stop |
| `max_tool_calls` | int | `20` | Hard stop |
| `max_payload_bytes` | int | `65536` | Soft (logged) |
| `max_latency_per_call_ms` | float | `5000` | Soft (logged) |
| `max_agent_tokens` | int | `32768` | Soft (logged) |
| `max_cumulative_result_tokens` | int | `0` | 0 = unlimited |

### Presets

| Preset | Turns | Tool Calls | Tokens | Notes |
|---|---|---|---|---|
| `fast` | 5 | 10 | 4096 | Quick smoke tests |
| `standard` | 10 | 20 | 32768 | Default |
| `extended` | 20 | 50 | 65536 | Thorough evaluation |
| `constrained-4k` | 6 | 12 | 16384 | Result budget: 4096 tokens |
| `constrained-2k` | 6 | 12 | 16384 | Result budget: 2048 tokens |

Individual fields can override preset defaults:

```json
{
  "agent_budget": {
    "preset": "standard",
    "max_turns": 15
  }
}
```

## SWE-bench Configuration (`swebench`)

| Field | Type | Default | Description |
|---|---|---|---|
| `max_steps` | int | `30` | Maximum agent loop iterations |
| `timeout` | int | `1800` | Task timeout in seconds |
| `workers` | int | `6` | Parallel task workers |
| `layout_file` | string\|null | `null` | Path to layout JSON (for stack+heap) |
| `no_think_prefill` | bool | `false` | Suppress Qwen3 think mode |
| `prebuild_only` | bool | `false` | Build images only, skip agent runs |

## Infrastructure Configuration (`infra`)

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"local"` | `"local"` or `"modal"` |
| `container_cmd` | string | `"podman"` | Container runtime: `"podman"` or `"docker"` |
| `docker_host` | string\|null | `null` | Container socket URL |
| `ghcr_prefix` | string | `"ghcr.io/marklubin/swebench"` | Registry prefix for pre-built images |

## CLI Reference

### `synix-bench run`

```
Options:
  --suite              Required. "swebench" or "lens"
  --strategy/--adapter Strategy or adapter name
  --dataset            Path to dataset file
  --config             Path to config JSON (overrides other flags)
  --out                Output directory [default: results]
  --model              LLM model name
  --base-url           LLM API base URL
  --api-key-env        Env var name containing API key
  --seed               Random seed [default: 42]
  --instance-id        Specific SWE-bench instance
  --sample             Number of instances to sample
  --trials             Trials per config [default: 1]
  --max-steps          Max agent steps [default: 30]
  --timeout            Timeout in seconds [default: 1800]
  --workers            Parallel workers [default: 6]
  --budget             Budget preset [default: standard]
  --no-think-prefill   Suppress Qwen3 think mode
  --layout             Layout config path (stack+heap)
  --provider           LLM provider (mock, openai)
  --parallel-questions Concurrent LENS questions
  --cache-dir          LENS adapter cache dir
  -v/--verbose         Increase verbosity (use twice for debug)
```

### `synix-bench sweep`

```
Options:
  --config   Required. Path to sweep config JSON
  --out      Output directory [default: results]
  --workers  Parallel workers [default: 4]
  -v         Verbose
```

### `synix-bench smoke`

```
Options:
  --suite    Suite to smoke test [default: lens]
```

### `synix-bench list`

```
Arguments:
  WHAT       What to list: suites, strategies, adapters, metrics
```

### `synix-bench score`

```
Options:
  --results  Path to result JSON(s)
  --tier     Filter to specific metric tier (1, 2, or 3)
```

### `synix-bench report`

```
Options:
  --results  Path to results directory or file(s)
  --output   Output file path
  --format   Output format (html, json, markdown)
```

### `synix-bench compare`

```
Options:
  --results  Two or more result JSON files to compare
  --output   Output file path
```
