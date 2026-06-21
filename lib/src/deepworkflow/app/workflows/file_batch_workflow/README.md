# file_batch_workflow

A LangGraph workflow that processes files in batches using a Map → Plan/Execute/Reflect/Evaluate → Reduce pattern.

## Graph

```mermaid
graph TD
    resolve_globs_step --> |static| effort_static_step
    resolve_globs_step --> |auto| effort_analyze_auto_agent

    effort_static_step --> |agent| map_batches_agent
    effort_static_step --> |step| map_batches_step
    effort_analyze_auto_agent --> |agent| map_batches_agent
    effort_analyze_auto_agent --> |step| map_batches_step

    map_batches_agent --> validate_map_batches_step
    map_batches_step --> validate_map_batches_step

    validate_map_batches_step --> |fail| fail_step
    validate_map_batches_step --> |retry| map_batches_agent
    validate_map_batches_step --> |evaluate| evaluate_map_batches_agent
    validate_map_batches_step --> |plan| plan_batch_agent
    validate_map_batches_step --> |skip_plan| skip_batch_plan_step

    evaluate_map_batches_agent --> |plan| plan_batch_agent
    evaluate_map_batches_agent --> |skip_plan| skip_batch_plan_step
    evaluate_map_batches_agent --> |retry_or_fail| map_increment_retry_step

    map_increment_retry_step --> |retry| map_batches_agent
    map_increment_retry_step --> |fail| fail_step

    plan_batch_agent --> execute_batch_agent
    skip_batch_plan_step --> execute_batch_agent
    execute_batch_agent --> |reflect| reflect_batch_agent
    execute_batch_agent --> |skip| skip_reflect_batch_step

    skip_reflect_batch_step --> skip_evaluate_quality_step

    reflect_batch_agent --> |evaluate_convergence| evaluate_batch_convergence_agent
    reflect_batch_agent --> |evaluate| evaluate_batch_quality_agent
    reflect_batch_agent --> |skip| skip_evaluate_quality_step

    evaluate_batch_convergence_agent --> |repeat| increment_batch_repeat_step
    evaluate_batch_convergence_agent --> |evaluate| evaluate_batch_quality_agent
    evaluate_batch_convergence_agent --> |skip| skip_evaluate_quality_step

    increment_batch_repeat_step --> |plan| plan_batch_agent
    increment_batch_repeat_step --> |skip_plan| skip_batch_plan_step

    skip_evaluate_quality_step --> record_output_step

    evaluate_batch_quality_agent --> |pass| record_output_step
    evaluate_batch_quality_agent --> |retry_or_fail| increment_retry_step

    increment_retry_step --> |plan| plan_batch_agent
    increment_retry_step --> |skip_plan| skip_batch_plan_step
    increment_retry_step --> |max_retries_exceeded| check_max_retries_policy_step

    check_max_retries_policy_step --> |fail| fail_step
    check_max_retries_policy_step --> |record| record_output_step

    record_output_step --> |plan| plan_batch_agent
    record_output_step --> |skip_plan| skip_batch_plan_step
    record_output_step --> |consolidate_agent| reduce_consolidate_agent
    record_output_step --> |consolidate_step| reduce_consolidate_step

    reduce_consolidate_agent --> END
    reduce_consolidate_step --> END
    fail_step --> END
```
