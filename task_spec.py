#!/usr/bin/env python3
"""
task_spec.py — SINGLE SOURCE OF TRUTH for the "explain-databricks" slide task.

Both run_task.py (the runner) and grade_tasks.py (the grader) import from here so
the prompt, the task brief, and the grading contract can NEVER drift apart. This
mirrors the fairness rule from the sibling agent-ml/ benchmark: everything except
"the step that produces the artifact" is byte-identical across every matrix cell.

Nothing in this module has side effects — it is pure constants + functions.
"""
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

TASK_ID = "explain-databricks"
ARTIFACT = "slides.html"

# ---- The COMMON PROMPT — MUST stay byte-identical for every harness. --------
# Kept short on the CLI; the heavy task detail lives in instructions.txt, which
# the agent is told to read. Direct-fmapi concatenates COMMON_PROMPT + the full
# instructions text into one message, so a non-agent model gets the same brief.
COMMON_PROMPT = (
    "Read ./instructions.txt and write a slide deck to ./slides.html that "
    "explains what Databricks is. Produce ONE self-contained HTML file: all CSS "
    "inline in a <style> tag (or an allowed reveal.js CDN), no local build step, "
    "and it must render when opened directly via file:// with no network access "
    "required for the core content. Follow the exact format contract, slide "
    "count, and required topics described in instructions.txt. Iterate until "
    "./slides.html exists and is a valid, complete, self-contained deck."
)


def load_task(task_id: str = TASK_ID) -> dict:
    """Load the machine-readable task config (keywords + slide-count bounds)."""
    kw_path = PROJECT_DIR / "tasks" / task_id / "keywords.json"
    if not kw_path.exists():
        raise FileNotFoundError(
            f"task config missing: {kw_path} — expected tasks/{task_id}/keywords.json"
        )
    return json.loads(kw_path.read_text(encoding="utf-8"))


def load_brief(task_id: str = TASK_ID) -> str:
    """Load the human-readable task brief (tasks/<task_id>/brief.md)."""
    brief_path = PROJECT_DIR / "tasks" / task_id / "brief.md"
    if not brief_path.exists():
        raise FileNotFoundError(
            f"task brief missing: {brief_path} — expected tasks/{task_id}/brief.md"
        )
    return brief_path.read_text(encoding="utf-8")


def build_instructions(task_id: str = TASK_ID) -> str:
    """The full instructions.txt body written into every workdir. Identical for
    every harness. Composed from the human brief + the machine format contract so
    the two can never disagree."""
    cfg = load_task(task_id)
    brief = load_brief(task_id)
    lo, hi = cfg["slide_count"]["min"], cfg["slide_count"]["max"]
    topics = "\n".join(f"  - {t}" for t in cfg["required_topics"])

    return f"""# Task: {task_id}

{brief}

## Hard format contract (the grader checks these mechanically)
- Write EXACTLY ONE file: ./slides.html (in this working directory).
- It MUST be a single self-contained HTML document:
  - Starts with `<!DOCTYPE html>` and has one <html>, <head>, and <body>.
  - All CSS is INLINE in a <style> tag. Do NOT link external stylesheets over
    http(s). (reveal.js from a CDN is tolerated but discouraged — prefer a fully
    offline file. External refs are flagged as a warning by the grader.)
  - Renders correctly when opened via file:// with no network for core content.
- Slide structure: use EITHER
  - vanilla slides: each slide is an element with class="slide", OR
  - reveal.js: each slide is a <section> element.
  The grader counts whichever it finds.
- Slide count: between {lo} and {hi} slides (inclusive).
- Required topics — every deck MUST cover each of these (the grader does a
  keyword-coverage check on the rendered text):
{topics}

## Reminders
- Work offline for the core content — do not depend on fetching data at render time.
- Keep iterating until ./slides.html exists and satisfies the contract above.
"""
