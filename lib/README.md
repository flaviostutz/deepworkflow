# deepworkflow

A graph of agents tailored to process a large number of files without compromising reasoning quality. The general workflow is **map → plan → execute → reflect → [repeat loop] → quality judge → reduce**.

Two judges operate in the per-batch loop with distinct responsibilities:

- **Progress judge** (`evaluate_batch_progress_agent`) — lightweight check after each `plan → execute → reflect` pass. Decides whether meaningful progress was made and whether to loop back for another pass (only active when `batch_repeat_max > 0`). Does **not** evaluate final quality.
- **Quality judge** (`evaluate_batch_quality_agent`) — final check that runs **once** after all passes for a batch complete. Evaluates the overall quality of the result and decides whether to accept it or retry the batch.

Built on top of [deepagents](https://github.com/langchain-ai/deepagents) — a LangGraph-based ReAct agent framework with filesystem support. Exposed as a Python library (LangGraph subgraph embeddable in other applications) and as a standalone CLI with config file.

## Getting Started

```bash
pip install deepworkflow
```

## Usage

### As a library (LangGraph subgraph)

```python
from langchain_openai import ChatOpenAI

from deepworkflow import run_workflow, DeepWorkflowConfig
from deepworkflow.shared.types import JudgeVerdict, OnMaxRetriesExceeded, WriteOption

# model is a required factory: called with agent name, returns a BaseChatModel
config = DeepWorkflowConfig(
    workspace_dir="/path/to/workspace",
    task_instructions="Review each file for security issues",
    model=lambda _: ChatOpenAI(model="gpt-4o"),
    workspace_write_option=WriteOption.READ_ONLY,
    judge_min=JudgeVerdict.WARNING,
    judge_max_retries=2,
    judge_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    log_level=WorkflowLogLevel.INFO,   # optional: NONE (default in lib), INFO, TRACE
    # task_files=["src/**/*.py"],  # Omit to let the agent discover files
)

result = run_workflow(config)
print(result.output)
```

### As a CLI

```bash
deepworkflow --config mydeepworkflow.yml
```

Add `--loglevel` to control console verbosity:

```bash
deepworkflow --config mydeepworkflow.yml --loglevel info
```

Example `deepworkflow.yml`:

```yaml
workspace_dir: /path/to/workspace
task_instructions: "Review each file for security issues"
model:
  model: gpt-4o
  model_provider: openai
workspace_write_option: read-only
judge_min: WARNING
judge_max_retries: 2
judge_on_max_retries: continue
log_level: info   # optional: info (default in CLI), trace, none
```

## Model Factory

The `model` parameter is a **required factory** `Callable[[str], BaseChatModel]`. It is called once per agent with the agent's name, enabling per-agent model routing:

```python
from langchain.chat_models import init_chat_model

def model_factory(agent_name: str):
    if "evaluate" in agent_name:
        return init_chat_model("gpt-4o-mini", model_provider="openai")
    return init_chat_model("gpt-4o", model_provider="openai")

config = DeepWorkflowConfig(model=model_factory, ...)
```

Agent names: `map_batches_agent`, `evaluate_map_batches_agent`, `plan_batch_agent`, `execute_batch_agent`, `reflect_batch_agent`, `evaluate_batch_progress_agent`, `evaluate_batch_quality_agent`, `reduce_consolidate_agent`.

## task_files — Existing Files Only

`task_files` must always reference **existing files in the workspace**. These are the source files that will be split across batches for processing. The map agent resolves every path and will reject any entry that does not exist on disk.

Glob patterns and line-range suffixes are expanded against the workspace before batching:

```python
task_files=["src/**/*.py", "tests/**/*.py"]   # globs over existing files
task_files=["README.md:34-56"]                 # line range of an existing file
```

## Repeat Loop (`batch_repeat_max`)

By default (`batch_repeat_max=0`) each batch runs a single `plan → execute → reflect` pass before the **quality judge** (`evaluate_batch_quality_agent`) evaluates the result.

Setting `batch_repeat_max` to a positive integer enables a **progress-driven repeat loop**: after each reflect, the **progress judge** (`evaluate_batch_progress_agent`) asks the model whether the pass made meaningful, non-trivial progress. If yes, and the ceiling has not been reached, a fresh `plan → execute → reflect` pass runs for the same batch. The **quality judge** then runs once after all passes complete.

Files touched across all passes are accumulated and reported together in `BatchOutput`.

```python
config = DeepWorkflowConfig(
    ...
    batch_repeat_max=3,  # allow up to 3 extra passes per batch
)
```

YAML equivalent:

```yaml
batch_repeat_max: 3
```

> **Note**: Each repeat pass is fully independent — the agent starts a fresh session with no memory of previous passes. Use this feature for tasks designed for incremental multi-pass work (e.g., "implement as many items as possible per pass").

## Output Files in Batch Instructions

If the task involves **creating or modifying output files**, the map plan must declare the expected output file list explicitly so that all batches can coordinate without conflicts:

- **Shared outputs** (produced collectively by all batches) → list them in `task_overview`
- **Batch-specific outputs** (each batch is responsible for its own set) → list them in that batch's `batch_instructions`

The judge (`evaluate_map_batches_agent`) verifies that output files are declared and flags plans that leave batches without a clear list of files they are responsible for.

## Workflow Logging (`log_level`)

The `log_level` setting controls console verbosity during a run.

| Level | Output |
|-------|--------|
| `none` | No console output (default in lib) |
| `trace` | Every MLflow span as JSON (raw tracing) |
| `info` | Agent/route headers, in/out summaries, elapsed time, and a summary block (default in CLI) |

```python
from deepworkflow.shared.types import WorkflowLogLevel

config = DeepWorkflowConfig(
    ...
    log_level=WorkflowLogLevel.INFO,
)
```

YAML equivalent:

```yaml
log_level: info
```

CLI flag (overrides YAML):

```bash
deepworkflow --config deepworkflow.yml --loglevel info
```

Sample INFO output:
```
> map_batches_agent
  > elapsed: 5s
> check_map_verdict [pass]
> plan_batch_agent
  > elapsed: 3s
> execute_batch_agent
  > (out) 2 files written
  > elapsed: 12s
> reflect_batch_agent
  > elapsed: 4s
> check_verdict [pass]
> reduce_consolidate_agent
  > elapsed: 6s
Summary:
  result: OK
  quality judge: OK
  total time: 42s
  total quality retries: 0
  total progress retries: 0
  total files read: 3
  total files written: 0
  model invocations: 6
  models total: 12k/2k tokens (~US$ 0.00)
  model gpt-4o: 12k/2k tokens
```

For TRACE level, each span can be piped into `jq`:

```bash
deepworkflow --config deepworkflow.yml --loglevel trace | jq 'select(.name == "execute_batch_agent")'
```
