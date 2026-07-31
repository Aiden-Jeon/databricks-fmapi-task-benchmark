"""Task-centric benchmark toolkit: run one candidate of a task and grade candidates.

The public modules are:
  - task_spec:   the shared contract (COMMON_PROMPT, task loaders) — single source of truth.
  - run_task:    produce <task>/<candidate>/slides.html via a chosen harness/model.
  - grade_tasks: validate + render + grade a task's candidates, build a review gallery.

Console entry points (see pyproject.toml): `run-task` and `grade-task`.
"""

__all__ = ["task_spec", "run_task", "grade_tasks"]
