#!/usr/bin/env python3
"""
task_spec.py — SINGLE SOURCE OF TRUTH for a benchmark task.

Layout is task-centric: each task lives in its own directory at the repo root:

    <task_id>/
    ├── README.md            # task overview
    ├── TASK_DESCRIPTION.md   # canonical brief + hard format contract (the instructions)
    ├── keywords.json         # machine grading config (slide bounds + required topics)
    └── <candidate>/          # one output dir per compared model/candidate (opus, sol, glm, …)

Both run_task.py (the runner) and grade_tasks.py (the grader) import from here so the
prompt, the task description, and the grading contract can NEVER drift apart. This mirrors
the fairness rule from the sibling agent-ml/ benchmark: everything except "the step that
produces the artifact" is byte-identical across every candidate.

Nothing in this module has side effects — it is pure constants + functions.
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_TASK = "explain-databricks"
ARTIFACT = "slides.html"

# ---- The COMMON PROMPT — MUST stay byte-identical for every candidate. ------
# Kept short on the CLI; the heavy task detail lives in TASK_DESCRIPTION.md (written
# into each candidate dir as instructions.txt), which the agent is told to read.
# direct-fmapi concatenates COMMON_PROMPT + the full instructions text into one message,
# so a non-agent model gets the same brief.
COMMON_PROMPT = (
    "Read ./instructions.txt and write a slide deck to ./slides.html that "
    "explains what Databricks is. Produce ONE self-contained HTML file: all CSS "
    "inline in a <style> tag (or an allowed reveal.js CDN), no local build step, "
    "and it must render when opened directly via file:// with no network access "
    "required for the core content. Follow the exact format contract, slide "
    "count, and required topics described in instructions.txt. Iterate until "
    "./slides.html exists and is a valid, complete, self-contained deck."
)


def task_dir(task: str = DEFAULT_TASK) -> Path:
    """The directory holding a task's description, config, and candidate outputs."""
    return PROJECT_DIR / task


def load_task(task: str = DEFAULT_TASK) -> dict:
    """Load the machine-readable task config (keywords + slide-count bounds)."""
    kw_path = task_dir(task) / "keywords.json"
    if not kw_path.exists():
        raise FileNotFoundError(
            f"task config missing: {kw_path} — expected {task}/keywords.json"
        )
    return json.loads(kw_path.read_text(encoding="utf-8"))


def load_description(task: str = DEFAULT_TASK) -> str:
    """Load the canonical task description (TASK_DESCRIPTION.md). This is copied
    verbatim into each candidate's instructions.txt so every candidate reads the
    exact same brief."""
    desc_path = task_dir(task) / "TASK_DESCRIPTION.md"
    if not desc_path.exists():
        raise FileNotFoundError(
            f"task description missing: {desc_path} — expected {task}/TASK_DESCRIPTION.md"
        )
    return desc_path.read_text(encoding="utf-8")
