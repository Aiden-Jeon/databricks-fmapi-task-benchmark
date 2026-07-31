#!/usr/bin/env python3
"""
run_task.py — Drive ONE (harness x model) cell of the "explain-databricks" slide
benchmark, keeping everything except the "produce slides.html" step byte-identical.

Mirrors the sibling agent-ml/run_agent.py, generalized from a single agent axis to
a (harness x model) matrix, and with a `direct-fmapi` baseline harness that calls
the Databricks Foundation Model API directly (one OpenAI-compatible chat completion).

What it does:
  1. Creates an isolated per-cell workdir  runs/run_<harness>_<model>/
  2. Writes an IDENTICAL instructions.txt (task brief + hard format contract) and a
     prompt.txt (COMMON_PROMPT verbatim) — byte-identical across every harness.
  3. Produces ./slides.html by ONE of:
       - direct-fmapi : a single FMAPI chat completion (no agent, the raw baseline)
       - claude-code  : `claude -p` headless CLI via subprocess
       - codex        : `codex exec` non-interactive CLI via subprocess
       - omnigent     : Databricks' meta-harness CLI (build_omnigent_argv — OPEN ITEM)
     all under a shared wall-clock budget.
  4. Captures wall-clock time + exit status, checks for slides.html, writes run_meta.json.

This script does NOT grade. Use grade_tasks.py for that.

A `--manual` mode reuses the identical workdir/prompt setup for GENUINELY UI-only
agents (Databricks Playground, an IDE side-panel): a human is a pure "keyboard proxy"
who pastes prompt.txt and brackets the agent-active time with two Enter presses.

Usage:
  python run_task.py --harness direct-fmapi --model databricks-claude-sonnet-4-6
  python run_task.py --harness claude-code
  python run_task.py --harness codex --max-seconds 1200
  python run_task.py --harness omnigent --model databricks-claude-sonnet-4-6 --omnigent-inner claude
  python run_task.py --harness playground --model databricks-claude-sonnet-4-6 --manual
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import task_spec

PROJECT_DIR = Path(__file__).resolve().parent

HARNESSES = ("direct-fmapi", "claude-code", "codex", "omnigent")


def slug(harness: str, model: str | None) -> str:
    """Run-dir name  run_<harness>_<model>  with the model sanitized to a safe
    filesystem token. One helper so the runner and grader agree on paths."""
    model_tok = re.sub(r"[^A-Za-z0-9.-]+", "-", model) if model else "default"
    return f"run_{harness}_{model_tok}"


def prepare_workdir(harness: str, model: str | None, task: str) -> Path:
    """Create runs/run_<harness>_<model>/ with instructions.txt + prompt.txt.

    Unlike the Kaggle sibling there is no task data to copy, so this only writes the
    two text files. Both are byte-identical across every harness (fairness rule)."""
    workdir = PROJECT_DIR / "runs" / slug(harness, model)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    instructions = task_spec.build_instructions(task)
    (workdir / "instructions.txt").write_text(instructions, encoding="utf-8")
    # Write COMMON_PROMPT as bytes to avoid any text-mode newline translation, so
    # it is byte-identical to what the subprocess harnesses pass on the CLI.
    (workdir / "prompt.txt").write_bytes(task_spec.COMMON_PROMPT.encode("utf-8"))

    return workdir


def build_omnigent_argv(model: str, inner_harness: str, max_turns: int) -> list[str]:
    """OPEN ITEM — exact non-interactive omnigent CLI syntax is unconfirmed.

    Contract this MUST satisfy:
      - run headless / non-interactively (no TTY REPL)
      - carry COMMON_PROMPT byte-identically as the task
      - select the foundation model = `model` (databricks-* via the AI Gateway)
      - optionally select the inner harness omnigent drives
      - cwd is the workdir; the agent writes ./slides.html

    Best current guess (from the omnigent docs / local CLI): an `omni run` with
    --harness/--model overrides. FILL IN once the exact syntax is confirmed; until
    then, run omnigent via --manual (the workdir/prompt prep is identical, so there
    is zero code risk)."""
    raise NotImplementedError(
        "omnigent CLI syntax not yet wired — run with --manual for now.\n"
        "Guess: ['omni','run','agent.yaml','--harness',inner_harness,'--model',model] "
        "with COMMON_PROMPT injected as the agent prompt."
    )


def build_argv(harness: str, model: str | None, inner_harness: str,
               max_turns: int) -> list[str]:
    """Per-harness headless invocation. The PROMPT (COMMON_PROMPT) is identical;
    only the wrapper flags differ. cwd is set to the workdir by the caller."""
    prompt = task_spec.COMMON_PROMPT
    if harness == "claude-code":
        # `claude -p` = headless/print mode. Uses its configured/subscription model.
        return [
            "claude", "-p", prompt,
            "--dangerously-skip-permissions",
            "--max-turns", str(max_turns),
        ]
    if harness == "codex":
        # `codex exec` = non-interactive. Uses provider/model from ~/.codex. No
        # native turn cap; bounded by the wall-clock timeout instead.
        return [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            prompt,
        ]
    if harness == "omnigent":
        return build_omnigent_argv(model, inner_harness, max_turns)
    sys.exit(f"ERROR: harness {harness!r} has no subprocess argv (use run_direct_fmapi/manual)")


def extract_html(text: str) -> str:
    """Pull a self-contained HTML document out of a chat completion that may wrap it
    in prose or a ```html fence. Prefer the span from the first <!DOCTYPE/<html to
    the last </html>; else strip a fenced code block; else return the text as-is."""
    if not text:
        return ""
    lower = text.lower()
    start = lower.find("<!doctype")
    if start == -1:
        start = lower.find("<html")
    end = lower.rfind("</html>")
    if start != -1 and end != -1 and end > start:
        return text[start:end + len("</html>")]
    # fenced block fallback: ```html ... ```  or  ``` ... ```
    m = re.search(r"```(?:html)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def base_meta(harness, model, task, workdir, argv, mode) -> dict:
    """Pre-initialized meta dict — a superset of the agent-ml schema so the grader
    can read every harness uniformly. Callers fill in the run-outcome fields."""
    return {
        "harness": harness,
        "model": model,                 # requested model (may be None)
        "effective_model": None,        # what actually ran (agents use their own)
        "task": task,
        "workdir": str(workdir),
        "mode": mode,                   # direct-fmapi | headless | manual
        "argv": argv,
        "common_prompt": task_spec.COMMON_PROMPT,
        "max_seconds": None,
        "max_turns": None,
        "wall_seconds": 0.0,
        "timed_out": False,
        "exit_code": None,
        "artifact_path": str(workdir / task_spec.ARTIFACT),
        "artifact_exists": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "started_at": None,
        "finished_at": None,
        "note": "",
    }


def write_meta(workdir: Path, meta: dict) -> None:
    """The one place run_meta.json is serialized."""
    (workdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[run_task] artifact exists: {meta['artifact_exists']}")
    print(f"[run_task] meta -> {workdir / 'run_meta.json'}")


def run_direct_fmapi(harness, model, task, workdir, max_seconds) -> dict:
    """The raw single-shot baseline: ONE OpenAI-compatible chat completion to FMAPI.

    Not apples-to-apples with the multi-turn agent harnesses (it gets one bounded
    completion and cannot self-correct) — labeled as a baseline in the grader."""
    from openai import OpenAI  # imported lazily so agent-only runs need no openai

    if not model or not model.startswith("databricks-"):
        sys.exit(
            "ERROR: --harness direct-fmapi requires --model starting with 'databricks-'.\n"
            "A bare 'claude-*'/'gpt-*' name routes to the vendor backend, NOT FMAPI."
        )
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if not host or not token:
        sys.exit("ERROR: set DATABRICKS_HOST and DATABRICKS_TOKEN for direct-fmapi.")

    artifact = workdir / task_spec.ARTIFACT
    instructions_text = (workdir / "instructions.txt").read_text(encoding="utf-8")
    content = task_spec.COMMON_PROMPT + "\n\n" + instructions_text

    meta = base_meta(harness, model, task, workdir, None, "direct-fmapi")
    meta["max_seconds"] = max_seconds
    meta["effective_model"] = model

    client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints",
                    timeout=max_seconds, max_retries=1)

    start = time.monotonic()
    meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=16000,
        )
        html = extract_html(resp.choices[0].message.content or "")
        artifact.write_text(html, encoding="utf-8")
        meta["exit_code"] = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            meta["completion_tokens"] = getattr(usage, "completion_tokens", None)
    except Exception as e:  # noqa: BLE001 — record any failure, never crash the run
        meta["note"] = f"FMAPI call failed ({type(e).__name__}: {e})"
        print(f"[run_task] ERROR: {meta['note']}")
    finally:
        meta["wall_seconds"] = round(time.monotonic() - start, 2)
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["artifact_exists"] = artifact.exists() and artifact.stat().st_size > 0

    return meta


def run_subprocess(harness, model, task, workdir, argv, max_seconds, max_turns) -> dict:
    """Headless subprocess path — factored from agent-ml/run_agent.py."""
    artifact = workdir / task_spec.ARTIFACT
    log_path = workdir / "agent_output.log"

    meta = base_meta(harness, model, task, workdir, argv, "headless")
    meta["max_seconds"] = max_seconds
    meta["max_turns"] = max_turns

    print(f"[run_task] argv={argv}")
    print(f"[run_task] launching (output tee -> {log_path}) ...")

    start = time.monotonic()
    meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                argv, cwd=str(workdir), stdout=logf,
                stderr=subprocess.STDOUT, timeout=max_seconds, text=True,
            )
        meta["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        meta["timed_out"] = True
    except FileNotFoundError:
        meta["note"] = f"CLI not found: {argv[0]!r} — is it installed / on PATH?"
        print(f"[run_task] ERROR: {meta['note']}")
    meta["wall_seconds"] = round(time.monotonic() - start, 2)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["artifact_exists"] = artifact.exists() and artifact.stat().st_size > 0

    print(f"[run_task] done in {meta['wall_seconds']:.1f}s  exit={meta['exit_code']}  "
          f"timed_out={meta['timed_out']}")
    if not meta["artifact_exists"]:
        print("[run_task] WARNING: no slides.html produced — see agent_output.log")
    return meta


def run_manual(harness, model, task, workdir) -> dict:
    """Semi-automated, UI-only run: a human is only a 'keyboard proxy'.

    Ported from agent-ml/run_agent.py:run_manual. The prompt and instructions are
    machine-prepared (byte-identical to headless runs); the human opens the workdir
    in the UI agent, pastes prompt.txt, and presses Enter to START/STOP so agent-
    active wall time is machine-bracketed. run_meta.json is ALWAYS written."""
    artifact = workdir / task_spec.ARTIFACT
    prompt_path = workdir / "prompt.txt"

    # Best-effort clipboard copy via pbcopy; NEVER fail if unavailable.
    clipboard = False
    if shutil.which("pbcopy"):
        try:
            subprocess.run(["pbcopy"], input=task_spec.COMMON_PROMPT, text=True,
                           check=True, timeout=10)
            clipboard = True
        except Exception:
            clipboard = False

    print("\n" + "=" * 72)
    print(f"[run_task] MANUAL (UI-only) mode — harness={harness!r} model={model!r}")
    print("You are a KEYBOARD PROXY only. Everything else is machine-controlled.")
    print("-" * 72)
    print("  1. Open your UI agent (Databricks Playground / IDE side-panel).")
    print(f"  2. Open ONLY this folder as the workspace:\n        {workdir}")
    print("  3. Paste the prompt into the agent:")
    print("        - it is on your clipboard now"
          f"{'' if clipboard else ' (pbcopy unavailable — copy it manually)'}")
    print(f"        - or copy it from: {prompt_path}")
    print(f"  4. Let the agent write its deck to ./slides.html\n        ({artifact})")
    print("-" * 72)
    print("Timing is human-bracketed: press Enter the moment you hit run, and")
    print("again the moment slides.html is finished.")
    print("=" * 72)

    meta = base_meta(harness, model, task, workdir, None, "manual")
    meta["effective_model"] = model
    aborted = False

    input(">>> Press Enter to START (right when you start the agent)... ")
    start = time.monotonic()
    meta["started_at"] = meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[run_task] STARTED at {meta['started_at']}")

    # INVARIANT: once START has happened, run_meta.json MUST ALWAYS be written.
    note = ("Timing is human-bracketed (semi-automated): a human pressed Enter at "
            "agent start/stop; all setup (workdir, prompt, instructions) was "
            "machine-prepared identically to the headless runs.")
    try:
        try:
            resp = input(">>> Press Enter to STOP (when slides.html is done), "
                         "or type 'q'+Enter to abort: ")
            aborted = resp.strip().lower() == "q"
        except (EOFError, KeyboardInterrupt):
            aborted = True
            print("\n[run_task] STOP interrupted (EOF/Ctrl-C) — aborting run.")
        meta["wall_seconds"] = round(time.monotonic() - start, 2)
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if aborted:
            note += " Run was ABORTED by the operator."
        print(f"[run_task] STOPPED at {meta['finished_at']}  "
              f"wall={meta['wall_seconds']:.1f}s{'  (ABORTED)' if aborted else ''}")

        meta["artifact_exists"] = artifact.exists() and artifact.stat().st_size > 0
        if not meta["artifact_exists"]:
            print("[run_task] WARNING: no slides.html produced in the workdir")

        # Aborted-run quarantine so the grader can NEVER pick up an aborted deck.
        if aborted and meta["artifact_exists"]:
            quarantined = workdir / "slides.aborted.html"
            try:
                artifact.replace(quarantined)
                meta["artifact_exists"] = False
                note += f" Artifact quarantined to {quarantined.name} (aborted run)."
                print(f"[run_task] quarantined slides.html -> {quarantined.name}")
            except OSError as e:
                note += (f" WARNING: failed to quarantine slides.html "
                         f"({type(e).__name__}: {e}); it may still be present.")
    finally:
        meta["note"] = note
        meta["aborted"] = aborted

    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run one (harness x model) cell of the explain-databricks slide benchmark.")
    ap.add_argument("--harness", required=True,
                    help=f"one of {HARNESSES} (or any name with --manual)")
    ap.add_argument("--model", default=None,
                    help="FMAPI model (databricks-*); required for direct-fmapi & omnigent")
    ap.add_argument("--manual", action="store_true",
                    help="semi-automated UI-only mode: no subprocess, human is a keyboard proxy")
    ap.add_argument("--max-seconds", type=int, default=900,
                    help="wall-clock cap (shared fairness budget); default 900s")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="turn cap for agents that support it (claude-code); default 40")
    ap.add_argument("--omnigent-inner", default="claude",
                    help="inner harness omnigent drives (default: claude)")
    ap.add_argument("--task", default=task_spec.TASK_ID, help="task id (default: explain-databricks)")
    args = ap.parse_args()

    if not args.manual and args.harness not in HARNESSES:
        ap.error(f"argument --harness: invalid choice {args.harness!r} "
                 f"(choose from {HARNESSES}); use --manual for an arbitrary UI-only agent")

    workdir = prepare_workdir(args.harness, args.model, args.task)
    print(f"[run_task] harness={args.harness}  model={args.model}")
    print(f"[run_task] workdir={workdir}")
    print(f"[run_task] budget={args.max_seconds}s  max_turns={args.max_turns}"
          f"  mode={'manual' if args.manual else args.harness}")

    if args.manual:
        meta = run_manual(args.harness, args.model, args.task, workdir)
    elif args.harness == "direct-fmapi":
        meta = run_direct_fmapi(args.harness, args.model, args.task, workdir, args.max_seconds)
    else:
        argv = build_argv(args.harness, args.model, args.omnigent_inner, args.max_turns)
        meta = run_subprocess(args.harness, args.model, args.task, workdir, argv,
                              args.max_seconds, args.max_turns)

    write_meta(workdir, meta)


if __name__ == "__main__":
    main()
