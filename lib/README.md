# deepworkflow

A graph of agents tailored to process a large number of files without compromising reasoning quality. The general workflow is **map → plan → execute → judge → reduce**.

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
    # task_files=["src/**/*.py"],  # Omit to let the agent discover files
)

result = run_workflow(config)
print(result.output)
```

### As a CLI

```bash
deepworkflow --config mydeepworkflow.yml
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

Agent names: `map_batches_agent`, `evaluate_map_batches_agent`, `plan_batch_agent`, `execute_batch_agent`, `reflect_batch_agent`, `evaluate_batch_agent`, `reduce_consolidate_agent`.

## task_files — Existing Files Only

`task_files` must always reference **existing files in the workspace**. These are the source files that will be split across batches for processing. The map agent resolves every path and will reject any entry that does not exist on disk.

Glob patterns and line-range suffixes are expanded against the workspace before batching:

```python
task_files=["src/**/*.py", "tests/**/*.py"]   # globs over existing files
task_files=["README.md:34-56"]                 # line range of an existing file
```

## Output Files in Batch Instructions

If the task involves **creating or modifying output files**, the map plan must declare the expected output file list explicitly so that all batches can coordinate without conflicts:

- **Shared outputs** (produced collectively by all batches) → list them in `task_overview`
- **Batch-specific outputs** (each batch is responsible for its own set) → list them in that batch's `batch_instructions`

The judge (`evaluate_map_batches_agent`) verifies that output files are declared and flags plans that leave batches without a clear list of files they are responsible for.
