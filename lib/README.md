# deepworkflow

A graph of agents tailored to process a large number of files without compromising reasoning quality. The general workflow is **resolve effort → map → [plan →] execute → reflect → [repeat loop] → evaluate quality → reduce**.

Two evaluators operate in the per-batch loop with distinct responsibilities:

- **Progress judge** (`evaluate_batch_convergence_agent`) — lightweight check after each `plan → execute → reflect` pass. Decides whether meaningful progress was made and whether to loop back for another pass (only active when `evaluate_batch_convergence_max_retries > 0` in the `EffortConfig`). Does **not** evaluate final quality.
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

from deepworkflow import run_workflow, DeepWorkflowConfig, resolveEffortConfig
from deepworkflow.shared.types import OnMaxRetriesExceeded, WriteOption

# model is a required factory: called with agent name, returns a BaseChatModel
config = DeepWorkflowConfig(
    workspace_dir="/path/to/workspace",
    task_instructions="Review each file for security issues",
    model=lambda _: ChatOpenAI(model="gpt-4o"),
    workspace_write_option=WriteOption.READ_ONLY,
    effort="custom",
    effort_config=resolveEffortConfig(5),          # level 1-10 preset
    evaluate_quality_min=JudgeLevel.WARNING,
    evaluate_quality_on_max_retries=OnMaxRetriesExceeded.CONTINUE,
    log_level=WorkflowLogLevel.INFO,               # optional: NONE (default in lib), INFO, DEBUG, TRACE
    # task_files=["src/**/*.py"],  # Omit to let the agent discover files
)

result = run_workflow(config)
print(result.output)
```

Use `effort="auto"` to let an agent pick the effort level automatically:

```python
config = DeepWorkflowConfig(
    workspace_dir="/path/to/workspace",
    task_instructions="Refactor all modules for consistency",
    model=lambda _: ChatOpenAI(model="gpt-4o"),
    workspace_write_option=WriteOption.WRITE_ANY,
    effort="auto",   # analyze_task_effort_agent determines the level
)
```

#### Clone workspace before running

Pass `clone_workspace_dir` to copy the workspace to a new directory before running. Agents will use the clone; the source is untouched. Raises `ValueError` if the target directory already exists.

```python
result = run_workflow(config, clone_workspace_dir="/tmp/workspace-clone")
```

### As a CLI

```bash
deepworkflow --config mydeepworkflow.yml
```

Add `--loglevel` to control console verbosity:

```bash
deepworkflow --config mydeepworkflow.yml --loglevel info
```

Save the consolidated output to a file (also printed to stdout):

```bash
deepworkflow --config mydeepworkflow.yml --output-file result.md
```

Run on a clone of the workspace, leaving the source untouched (fails if the target directory already exists):

```bash
deepworkflow --config mydeepworkflow.yml --clone-workspace-dir /tmp/workspace-clone
```

Example `deepworkflow.yml`:

```yaml
workspace_dir: /path/to/workspace
task_instructions: "Review each file for security issues"
model:
  model: gpt-4o
  model_provider: openai
workspace_write_option: read-only
effort: custom
effort_config:
  level: 5              # shorthand: resolveEffortConfig(5) preset
evaluate_quality_min: WARNING
evaluate_quality_on_max_retries: continue
log_level: info         # optional: info (default in CLI), debug, trace, none
```

## Effort Configuration

The `effort` field controls how much computational work the workflow performs — from a single LLM pass to fully agentic multi-step execution with evaluation and retries.

### `effort="custom"` — explicit `EffortConfig`

Set every knob directly using an `EffortConfig` dataclass, or use `resolveEffortConfig(level)` for a preset:

```python
from deepworkflow import resolveEffortConfig
from deepworkflow.shared.types import EffortConfig

# Level preset (recommended)
effort_config = resolveEffortConfig(5)   # moderate agentic workflow

# Or fully custom
effort_config = EffortConfig(
    map_batches_mode="agent",             # "agent" or "static"
    max_batches=None,                     # no batch limit
    max_files_per_batch=None,             # no per-batch file limit
    evaluate_map_max_retries=2,           # map quality evaluation retries
    skip_batch_plan=False,                # run plan_batch_agent before each execute
    evaluate_batch_convergence_max_retries=1,  # repeat-loop passes per batch
    evaluate_batch_quality_max_retries=2, # quality retries per batch
    consolidate_mode="agent",             # "agent" or "static"
)
```

### `effort="auto"` — agent-determined level

The `analyze_task_effort_agent` reads the task instructions and a sample of workspace files, then selects a level from 1–10 and calls `resolveEffortConfig` internally. No `effort_config` field is required.

### Level presets (`resolveEffortConfig`)

| Level | Map mode | Consolidate | Plan agent | Eval retries | Notes |
|-------|----------|-------------|------------|--------------|-------|
| 1 | static | static | skipped | 0 | Single batch, fully mechanical |
| 2–3 | static | static | skipped | low | Multi-batch static, no evaluation |
| 4–5 | agent | agent | skipped | medium | LLM batching + quality checks |
| 6–7 | agent | agent | enabled | medium | Planning step added |
| 8–9 | agent | agent | enabled | high | Multiple convergence passes |
| 10 | agent | agent | enabled | 10 | Maximum safeguards |

### `EffortConfig` fields

| Field | Default | Description |
|-------|---------|-------------|
| `map_batches_mode` | `"agent"` | `"agent"` uses LLM to split files; `"static"` uses sequential grouping |
| `max_batches` | `None` | Maximum number of batches; `None` = no limit |
| `max_files_per_batch` | `None` | Required for `static` mode unless `max_batches=1` |
| `evaluate_map_max_retries` | `1` | Retries for map evaluation; `0` skips LLM map evaluation |
| `skip_batch_plan` | `False` | Skip `plan_batch_agent`; inject planning instruction into execute instead |
| `evaluate_batch_convergence_max_retries` | `0` | Extra plan→execute→reflect passes per batch (repeat loop) |
| `evaluate_batch_quality_max_retries` | `1` | Quality retries per batch; `0` skips quality evaluation |
| `consolidate_mode` | `"agent"` | `"agent"` uses LLM to consolidate; `"static"` formats outputs as markdown |

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

Agent names: `analyze_task_effort_agent`, `map_batches_agent`, `evaluate_map_batches_agent`, `plan_batch_agent`, `execute_batch_agent`, `reflect_batch_agent`, `evaluate_batch_convergence_agent`, `evaluate_batch_quality_agent`, `reduce_consolidate_agent`.

## task_files — Existing Files Only

`task_files` must always reference **existing files in the workspace**. These are the source files that will be split across batches for processing. The map agent resolves every path and will reject any entry that does not exist on disk.

Glob patterns and line-range suffixes are expanded against the workspace before batching:

```python
task_files=["src/**/*.py", "tests/**/*.py"]   # globs over existing files
task_files=["README.md:34-56"]                 # line range of an existing file
```

## Repeat Loop (`evaluate_batch_convergence_max_retries`)

By default (`evaluate_batch_convergence_max_retries=0` in `EffortConfig`) each batch runs a single `plan → execute → reflect` pass before **evaluate quality** evaluates the result.

Setting `evaluate_batch_convergence_max_retries` to a positive integer enables a **progress-driven repeat loop**: after each reflect, the `evaluate_batch_convergence_agent` asks the model whether the pass made meaningful, non-trivial progress. If yes, and the ceiling has not been reached, a fresh `plan → execute → reflect` pass runs for the same batch. **Evaluate quality** then runs once after all passes complete.

Files touched across all passes are accumulated and reported together in `BatchOutput`.

```python
config = DeepWorkflowConfig(
    ...
    effort="custom",
    effort_config=EffortConfig(
        evaluate_batch_convergence_max_retries=3,  # allow up to 3 extra passes per batch
        # ... other fields
    ),
)
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
| `info` | Agent/route headers, in/out summaries, elapsed time, and a summary block (default in CLI) |
| `debug` | Like `info` but prints full LLM-generated text (plans, outputs, evaluations) without truncation |
| `trace` | Every MLflow span as JSON (raw tracing) |

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
> analyze_task_effort_agent [level=5]
  > elapsed: 3s
> map_batches_agent
  > elapsed: 5s
> validate_map_batches_step [pass]
> plan_batch_agent
  > elapsed: 3s
> execute_batch_agent
  > (out) 2 files written
  > elapsed: 12s
> reflect_batch_agent
  > elapsed: 4s
> check_verdict [pass]
> reduce_consolidate_agent
  > (out) output:
  > (out) The refactored files now use consistent error handling across all modules.
  > (out) Summary of changes: ...
  > (out) 3 files read; 2 files written
  > elapsed: 6s
Summary:
  result: OK
  evaluate quality: OK
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
