"""databricks-app-generation benchmark package.

Suite layout (suite root = the databricks-app-generation/ directory):
    suite.json                     tier composition, weights, gate rule, efficiency axis
    <tier>/TASK_DESCRIPTION.md     canonical brief per tier (copied as instructions.txt)
    <tier>/test_cases.json         machine grading config per tier
    <tier>/<candidate>/app/        one output app per candidate per tier

Shares the skeleton of html-slide-generation/src/benchmark: task_spec is the single
source of truth; run_task swaps only the "produce the artifact" step; grade_tasks
applies one identical grader to every candidate.
"""
