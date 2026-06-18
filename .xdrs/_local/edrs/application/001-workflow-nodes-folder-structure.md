---
name: local-edr-001-workflow-nodes-folder-structure
description: Override for agentme-edr-020 rule 07-workflow-structure. Use a nodes/ sub-package instead of a single agents.py file for workflow node implementations.
apply-to: deepworkflow project — lib/src/deepworkflow/app/workflows/
valid-from: 2026-06-06
overrides: agentme-edr-020 rule 07-workflow-structure (agents.py file name)
---

# local-edr-001: Workflow nodes folder structure

## Context and Problem Statement

[agentme-edr-020](../../../agentme/edrs/application/020-ai-workflow-development-standards.md) rule `07-workflow-structure` prescribes an `agents.py` file inside each `app/workflows/<workflow>/` folder to hold all deepagents agent node definitions.

The `file_batch_workflow` has seven agent nodes and several deterministic step nodes, giving a total of more than ten node functions. A single `agents.py` file would exceed the 400-line file limit set by [agentme-edr-002](../../../agentme/edrs/principles/002-coding-best-practices.md) rule `01-keep-files-short`.

## Decision Outcome

**Use a `nodes/` sub-package instead of a flat `agents.py` file.** Each node (agent or step) lives in its own module inside `nodes/`, named after the node function it exports.

### Layout

```text
app/workflows/<workflow>/
  graph.py
  states.py
  routes.py
  nodes/
    __init__.py
    map_batches_agent.py
    evaluate_map_batches_agent.py
    plan_batch_agent.py
    execute_batch_agent.py
    reflect_batch_agent.py
    evaluate_batch_progress_agent.py  # evaluate_batch_progress_agent: checks whether meaningful progress was made each pass (repeat loop only)
    evaluate_batch_quality_agent.py   # evaluate_batch_quality_agent: final quality check after all passes complete
    reduce_consolidate_agent.py
    resolve_globs_step.py
```

### Rules

- Each module in `nodes/` MUST export exactly one node function whose name matches the module filename.
- Step nodes (deterministic, no LLM) MUST use the `_step` suffix; agent nodes (deepagents) MUST use the `_agent` suffix, consistent with [agentme-edr-020](../../../agentme/edrs/application/020-ai-workflow-development-standards.md) rule `09-node-naming-conventions`.
- `nodes/__init__.py` MAY expose shared helpers (e.g. `parse_evaluate_output`) but MUST NOT re-export node functions — callers import directly from the sub-module.

## References

- [agentme-edr-020](../../../agentme/edrs/application/020-ai-workflow-development-standards.md) — AI workflow development standards (overridden rule: `07-workflow-structure`)
- [agentme-edr-002](../../../agentme/edrs/principles/002-coding-best-practices.md) — Coding best practices (rule `01-keep-files-short`)
