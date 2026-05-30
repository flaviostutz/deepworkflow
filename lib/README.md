# deepworkflow

A graph of agents tailored to process a large number of files without compromising reasoning quality. The general workflow is **map → plan → execute → judge → reduce**.

Built on top of [deepagents](https://github.com/langchain-ai/deepagents) — a LangGraph-based ReAct agent framework with filesystem support. Exposed as a Python library (LangGraph subgraph embeddable in other applications) and as a standalone CLI with config file.

## Installation

```bash
pip install deepworkflow
```

## Usage

### As a library (LangGraph subgraph)

```python
from deepworkflow import run_workflow, WorkflowConfig

config = WorkflowConfig(
    workspace_dir="/path/to/workspace",
    task_instructions=["Review each file for security issues"],
    task_files_write_option="read-only",
    judge_minimum="WARNING",
    judge_max_retries=2,
    on_max_retries_exceeded="continue",
)

result = run_workflow(config)
print(result)
```

### As a CLI

```bash
deepworkflow --config workflow.yaml
```
