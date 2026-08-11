#!/usr/bin/env python3
"""
task_spec.py — SINGLE SOURCE OF TRUTH for the databricks-app-generation suite.

Suite-centric layout (vs. the single-task layout of html-slide-generation):

    databricks-app-generation/          <- SUITE dir (this is what lives under the repo root)
    ├── suite.json                      # tier composition + weights + gate + efficiency
    ├── pricing.json                    # model -> $/1M tokens (cost axis)
    ├── tier1-gate/
    │   ├── TASK_DESCRIPTION.md         # canonical brief (copied to instructions.txt)
    │   ├── test_cases.json             # machine grading config
    │   ├── ground_truth.sql            # named GT queries, recomputed at grading time
    │   └── <candidate>/                # one output dir per candidate (opus, sol, glm, …)
    │       └── app/                    #   the artifact: app.py + app.yaml + requirements.txt
    ├── tier2-core/  …
    └── tier3-differentiator/
        └── legacy_app/                 # copied into each candidate workdir (repair task)

Resolution mirrors the sibling benchmark: dirs are found relative to the BENCHMARK ROOT
(cwd by default, override with BENCHMARK_ROOT). Both run_task.py and grade_tasks.py import
from here so prompt / instructions / grading can never drift apart.

Nothing here has side effects — pure constants + loaders.
"""
import json
import os
import re
from pathlib import Path

DEFAULT_SUITE = "databricks-app-generation"
ARTIFACT_DIR = "app"                      # the artifact is a directory, not a single file
REQUIRED_FILES = ("app.py", "app.yaml", "requirements.txt")
TIERS = ("tier1-gate", "tier2-core", "tier3-differentiator")

# Candidate -> FMAPI model mapping (same convention as html-slide-generation).
CANDIDATE_MODELS = {
    "opus": "databricks-claude-opus-4-8",
    "sol": "databricks-gpt-5-6-sol",
    "glm": "databricks-glm-5-2",
}
PI_CANDIDATE_MODELS = {
    "opus": ("databricks-claude", "system.ai.claude-opus-5"),
    "sol": ("databricks-openai", "system.ai.gpt-5-5"),
}

# ---- The COMMON PROMPT — byte-identical for every candidate and every tier. -------
# Tier-specific detail lives in each tier's TASK_DESCRIPTION.md (-> instructions.txt).
# The FILE-marker output requirement makes a single-shot (direct-fmapi) response
# machine-splittable into files; agent harnesses simply write the files and the runner
# ignores their chat output.
COMMON_PROMPT = (
    "Read ./instructions.txt and build the Databricks App it describes. Create the "
    "app source under ./app/ in this working directory: app/app.py, app/app.yaml, "
    "and app/requirements.txt, exactly as the format contract in instructions.txt "
    "specifies. If ./legacy_app/ exists here, it is the starting codebase you must "
    "repair and extend per the instructions. Iterate until ./app/ exists and "
    "satisfies the contract.\n\n"
    "Output requirement (applies to your FINAL response): emit every file in full, "
    "each preceded by a marker line of the exact form `=== FILE: app/<name> ===` and "
    "followed by the complete file content, with NO other prose, explanation, or "
    "markdown fences. Example:\n"
    "=== FILE: app/app.yaml ===\n"
    "command: [\"streamlit\", \"run\", \"app.py\"]\n"
    "=== FILE: app/app.py ===\n"
    "..."
)

# Match `=== FILE: <path> ===` markers. The marker normally sits on its own line, but
# reasoning models (e.g. GLM) can emit a `</think>` tag or other text immediately before
# the first marker on the SAME line (e.g. `...code.</think>=== FILE: app/app.yaml ===`),
# which a strict `^...$` anchor would miss — dropping that file. So we don't anchor to
# line start; `=== FILE:` is distinctive enough to avoid false positives. We still anchor
# the trailing `===` to end-of-line so a path can't swallow following content.
FILE_MARKER_RE = re.compile(r"===\s*FILE:\s*(?P<path>[^=\n]+?)\s*===\s*$", re.MULTILINE)


def benchmark_root() -> Path:
    return Path(os.environ.get("BENCHMARK_ROOT", Path.cwd())).resolve()


def suite_dir(suite: str = DEFAULT_SUITE) -> Path:
    return benchmark_root() / suite


def tier_dir(tier: str, suite: str = DEFAULT_SUITE) -> Path:
    return suite_dir(suite) / tier


def load_suite(suite: str = DEFAULT_SUITE) -> dict:
    p = suite_dir(suite) / "suite.json"
    if not p.exists():
        raise FileNotFoundError(
            f"suite config missing: {p} — run from the repo root or set BENCHMARK_ROOT.")
    return json.loads(p.read_text(encoding="utf-8"))


def load_pricing(suite: str = DEFAULT_SUITE) -> dict:
    p = suite_dir(suite) / "pricing.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _resolve_same_as(cfg: dict, tier: str, suite: str) -> dict:
    """test_cases.json blocks may point at another tier's block via {'same_as': <relpath>}."""
    out = dict(cfg)
    for key, val in cfg.items():
        if isinstance(val, dict) and "same_as" in val:
            ref = (tier_dir(tier, suite) / val["same_as"]).resolve()
            ref_cfg = json.loads(ref.read_text(encoding="utf-8"))
            out[key] = ref_cfg.get(key, {})
    return out


def load_tier(tier: str, suite: str = DEFAULT_SUITE) -> dict:
    p = tier_dir(tier, suite) / "test_cases.json"
    if not p.exists():
        raise FileNotFoundError(f"tier config missing: {p}")
    return _resolve_same_as(json.loads(p.read_text(encoding="utf-8")), tier, suite)


def load_description(tier: str, suite: str = DEFAULT_SUITE) -> str:
    p = tier_dir(tier, suite) / "TASK_DESCRIPTION.md"
    if not p.exists():
        raise FileNotFoundError(f"task description missing: {p}")
    return p.read_text(encoding="utf-8")


def load_ground_truth_queries(tier: str, suite: str = DEFAULT_SUITE) -> dict[str, str]:
    """Parse ground_truth.sql into {query_name: sql}. Queries are delimited by
    `-- gt_<name>` comment headers and terminated by `;`. The path may be redirected
    by test_cases.json's `ground_truth_sql` (e.g. tier3 reuses tier1's)."""
    cfg = load_tier(tier, suite)
    rel = cfg.get("ground_truth_sql", "ground_truth.sql")
    p = (tier_dir(tier, suite) / rel).resolve()
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    current_name, buf = None, []
    for line in text.splitlines():
        m = re.match(r"--\s*(gt_\w+)", line)
        if m:
            if current_name and buf:
                queries[current_name] = "\n".join(buf).strip().rstrip(";")
            current_name, buf = m.group(1), []
        elif current_name is not None and not line.strip().startswith("--"):
            buf.append(line)
    if current_name and buf:
        queries[current_name] = "\n".join(buf).strip().rstrip(";")
    return {k: v for k, v in queries.items() if v}
