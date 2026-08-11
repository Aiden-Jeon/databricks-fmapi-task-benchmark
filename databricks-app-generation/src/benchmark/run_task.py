#!/usr/bin/env python3
"""
run_task.py — Drive ONE candidate through one tier (or all tiers) of the suite,
keeping everything except the "produce ./app/" step byte-identical across candidates.

Adapted from html-slide-generation/src/benchmark/run_task.py. Differences:
  - --tier <tier|all>: outputs go to <suite>/<tier>/<candidate>/
  - the artifact is a DIRECTORY (app/app.py + app.yaml + requirements.txt)
  - direct-fmapi splits the single completion into files via `=== FILE: ... ===` markers
  - tier3 copies legacy_app/ into the workdir before the run
  - per-tier wall-clock budget from test_cases.json (timeout == that tier failed)
  - token usage recorded per run for the cost axis (tokens_source notes provenance)

Usage:
  run-task --tier tier1-gate --candidate opus --harness direct-fmapi
  run-task --tier all        --candidate opus --harness claude-code
  # equivalently:  python -m benchmark.run_task ...
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

from benchmark import fmapi_auth, task_spec

HARNESSES = ("direct-fmapi", "claude-code", "codex", "pi", "omnigent")
UCODE_PI_AGENT_DIR = Path.home() / ".ucode" / "pi-home" / ".pi" / "agent"


def candidate_dir(suite: str, tier: str, candidate: str) -> Path:
    return task_spec.tier_dir(tier, suite) / candidate


def prepare_workdir(suite: str, tier: str, candidate: str) -> Path:
    """Create <suite>/<tier>/<candidate>/ with byte-identical instructions.txt +
    prompt.txt; for tier3, also copy legacy_app/ (the codebase to repair)."""
    workdir = candidate_dir(suite, tier, candidate)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    (workdir / "instructions.txt").write_text(
        task_spec.load_description(tier, suite), encoding="utf-8")
    (workdir / "prompt.txt").write_bytes(task_spec.COMMON_PROMPT.encode("utf-8"))

    legacy_src = task_spec.tier_dir(tier, suite) / "legacy_app"
    if legacy_src.exists():
        shutil.copytree(legacy_src, workdir / "legacy_app")

    return workdir


# --------------------------------------------------------- multi-file extraction ---
def extract_files(text: str) -> dict[str, str]:
    """Split a single completion into {relative_path: content} using
    `=== FILE: <path> ===` marker lines. Strips one optional markdown fence wrapped
    around a file body. Paths are sanitized to stay under app/."""
    files: dict[str, str] = {}
    if not text:
        return files
    # Drop reasoning blocks some models emit before the files (e.g. GLM's
    # `<think>...</think>`). Left in, the closing tag can share a line with the first
    # marker and the text can leak into a body; removing it keeps extraction clean.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "").replace("<think>", "")
    matches = list(task_spec.FILE_MARKER_RE.finditer(text))
    for i, m in enumerate(matches):
        raw_path = m.group("path").strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip("\n")
        fence = re.match(r"^```[\w-]*\n(.*)\n```\s*$", body, re.DOTALL)
        if fence:
            body = fence.group(1)
        # sanitize: forbid absolute paths / traversal; force under app/
        path = raw_path.lstrip("/").replace("\\", "/")
        if ".." in path.split("/"):
            continue
        if not path.startswith(f"{task_spec.ARTIFACT_DIR}/"):
            path = f"{task_spec.ARTIFACT_DIR}/{Path(path).name}"
        files[path] = body + "\n"
    return files


def artifact_complete(workdir: Path) -> bool:
    app = workdir / task_spec.ARTIFACT_DIR
    return all((app / f).exists() and (app / f).stat().st_size > 0
               for f in task_spec.REQUIRED_FILES)


# ------------------------------------------------------------------ harness argv ---
def build_omnigent_argv(model, inner_harness, max_turns) -> list[str]:
    """OPEN ITEM (inherited from the sibling benchmark) — exact non-interactive
    omnigent CLI syntax unconfirmed; run with --manual until wired."""
    raise NotImplementedError("omnigent CLI syntax not yet wired — use --manual.")


def build_argv(harness, model, inner_harness, max_turns, pi_provider=None) -> list[str]:
    prompt = task_spec.COMMON_PROMPT
    if harness == "claude-code":
        return ["claude", "-p", prompt, "--dangerously-skip-permissions",
                "--max-turns", str(max_turns)]
    if harness == "codex":
        return ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check", prompt]
    if harness == "pi":
        argv = ["pi", "--print", "--no-session"]
        if pi_provider:
            argv += ["--provider", pi_provider]
        if model:
            argv += ["--model", model]
        argv.append(prompt)
        return argv
    if harness == "omnigent":
        return build_omnigent_argv(model, inner_harness, max_turns)
    sys.exit(f"ERROR: harness {harness!r} has no subprocess argv")


def pi_env() -> dict:
    if not (UCODE_PI_AGENT_DIR / "models.json").exists():
        sys.exit(f"ERROR: pi not configured ({UCODE_PI_AGENT_DIR}/models.json missing). "
                 "Run: ucode configure --agent pi --skip-validate")
    return {"PI_CODING_AGENT_DIR": str(UCODE_PI_AGENT_DIR)}


# ------------------------------------------------------------------------- meta ---
def base_meta(suite, tier, candidate, harness, model, workdir, argv, mode) -> dict:
    return {
        "suite": suite, "tier": tier, "candidate": candidate,
        "harness": harness, "model": model, "effective_model": None,
        "workdir": str(workdir), "mode": mode, "argv": argv,
        "common_prompt": task_spec.COMMON_PROMPT,
        "max_seconds": None, "max_turns": None,
        "wall_seconds": 0.0, "timed_out": False, "exit_code": None,
        "artifact_dir": str(workdir / task_spec.ARTIFACT_DIR),
        "artifact_complete": False,
        "prompt_tokens": None, "completion_tokens": None,
        "tokens_source": "unavailable",   # api-usage | session-log | unavailable
        "started_at": None, "finished_at": None, "note": "",
    }


def write_meta(workdir: Path, meta: dict) -> None:
    (workdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[run_task] artifact complete: {meta['artifact_complete']}")
    print(f"[run_task] meta -> {workdir / 'run_meta.json'}")


# ----------------------------------------------------------------- run: fmapi ----
def run_direct_fmapi(suite, tier, candidate, harness, model, workdir, max_seconds) -> dict:
    """Raw single-shot baseline: ONE chat completion, split into files afterwards.
    Not apples-to-apples with multi-turn agents — labeled a baseline in the grader."""
    from openai import OpenAI

    if not model or not model.startswith("databricks-"):
        sys.exit("ERROR: direct-fmapi needs a databricks-* model "
                 "(bare claude-*/gpt-* routes to the vendor backend).")
    host, token, source = fmapi_auth.resolve_host_token()
    if not host or not token:
        sys.exit("ERROR: could not resolve FMAPI credentials (env or ucode).")
    print(f"[run_task] FMAPI host={host} (auth source: {source})")

    instructions_text = (workdir / "instructions.txt").read_text(encoding="utf-8")
    content = task_spec.COMMON_PROMPT + "\n\n" + instructions_text
    if (workdir / "legacy_app").exists():
        # tier3: a non-agent model cannot read files — inline the legacy codebase.
        parts = ["\n\n--- LEGACY CODEBASE (./legacy_app/) ---"]
        for f in sorted((workdir / "legacy_app").rglob("*")):
            if f.is_file():
                rel = f.relative_to(workdir)
                parts.append(f"\n=== FILE: {rel} ===\n{f.read_text(encoding='utf-8')}")
        content += "".join(parts)

    meta = base_meta(suite, tier, candidate, harness, model, workdir, None, "direct-fmapi")
    meta["max_seconds"] = max_seconds
    meta["effective_model"] = model

    client = OpenAI(api_key=token, base_url=f"{host}/serving-endpoints",
                    timeout=max_seconds, max_retries=1)
    start = time.monotonic()
    meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}],
            max_tokens=32000)
        raw = resp.choices[0].message.content or ""
        (workdir / "raw_completion.txt").write_text(raw, encoding="utf-8")
        files = extract_files(raw)
        for rel, body in files.items():
            dest = workdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        meta["exit_code"] = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            meta["completion_tokens"] = getattr(usage, "completion_tokens", None)
            meta["tokens_source"] = "api-usage"
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            meta["note"] = "completion hit max_tokens — app files may be truncated"
            print(f"[run_task] WARNING: {meta['note']}")
        if not files:
            meta["note"] = (meta["note"] + "; " if meta["note"] else "") + \
                "no FILE markers found in completion (see raw_completion.txt)"
    except Exception as e:  # noqa: BLE001
        meta["note"] = f"FMAPI call failed ({type(e).__name__}: {e})"
        print(f"[run_task] ERROR: {meta['note']}")
    finally:
        meta["wall_seconds"] = round(time.monotonic() - start, 2)
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["artifact_complete"] = artifact_complete(workdir)
    return meta


# ------------------------------------------------------------- run: subprocess ---
def harvest_session_tokens(log_path: Path) -> tuple[int | None, int | None]:
    """Best-effort token harvest from harness output logs (claude-code/codex print
    usage summaries in some versions). Returns (prompt, completion) or (None, None)."""
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"input[_ ]tokens\D{0,10}(\d[\d,]*)", text, re.IGNORECASE)
    n = re.search(r"output[_ ]tokens\D{0,10}(\d[\d,]*)", text, re.IGNORECASE)
    p = int(m.group(1).replace(",", "")) if m else None
    c = int(n.group(1).replace(",", "")) if n else None
    return p, c


def run_subprocess(suite, tier, candidate, harness, model, workdir, argv, max_seconds,
                   max_turns, env_overlay=None) -> dict:
    log_path = workdir / "agent_output.log"
    meta = base_meta(suite, tier, candidate, harness, model, workdir, argv, "headless")
    meta["max_seconds"] = max_seconds
    meta["max_turns"] = max_turns

    child_env = {**os.environ, **env_overlay} if env_overlay else None
    print(f"[run_task] argv={argv}")
    print(f"[run_task] launching (output tee -> {log_path}) ...")

    start = time.monotonic()
    meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(argv, cwd=str(workdir), stdout=logf,
                                  stderr=subprocess.STDOUT, timeout=max_seconds,
                                  text=True, env=child_env)
        meta["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        meta["timed_out"] = True   # timeout == this tier failed (suite rule)
    except FileNotFoundError:
        meta["note"] = f"CLI not found: {argv[0]!r}"
        print(f"[run_task] ERROR: {meta['note']}")
    meta["wall_seconds"] = round(time.monotonic() - start, 2)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["artifact_complete"] = artifact_complete(workdir)

    p, c = harvest_session_tokens(log_path)
    if p or c:
        meta["prompt_tokens"], meta["completion_tokens"] = p, c
        meta["tokens_source"] = "session-log"

    print(f"[run_task] done in {meta['wall_seconds']:.1f}s exit={meta['exit_code']} "
          f"timed_out={meta['timed_out']}")
    if not meta["artifact_complete"]:
        print("[run_task] WARNING: app/ incomplete — see agent_output.log")
    return meta


# ------------------------------------------------------------------------ main ---
def run_one_tier(args, tier: str) -> dict:
    cfg = task_spec.load_tier(tier, args.suite)
    budget_s = args.max_seconds or int(cfg.get("budget_minutes", 30)) * 60
    workdir = prepare_workdir(args.suite, tier, args.candidate)
    print(f"[run_task] suite={args.suite} tier={tier} candidate={args.candidate}")
    print(f"[run_task] harness={args.harness} model={args.model} budget={budget_s}s")

    if args.harness == "direct-fmapi":
        meta = run_direct_fmapi(args.suite, tier, args.candidate, args.harness,
                                args.model, workdir, budget_s)
    else:
        argv = build_argv(args.harness, args.model, args.omnigent_inner, args.max_turns,
                          pi_provider=args.pi_provider)
        env_overlay = pi_env() if args.harness == "pi" else None
        meta = run_subprocess(args.suite, tier, args.candidate, args.harness, args.model,
                              workdir, argv, budget_s, args.max_turns, env_overlay)
        if args.harness == "pi":
            meta["effective_model"] = f"{args.pi_provider}/{args.model}"

    write_meta(workdir, meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run one candidate through one tier (or all) of the app suite.")
    ap.add_argument("--suite", default=task_spec.DEFAULT_SUITE)
    ap.add_argument("--tier", default="all",
                    help=f"one of {task_spec.TIERS} or 'all' (default: all; "
                         "'all' runs 1->2->3 and stops if the gate tier fails)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--harness", required=True, choices=HARNESSES)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="override the tier's budget_minutes (fairness: avoid)")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--omnigent-inner", default="claude")
    ap.add_argument("--pi-provider", default=None)
    args = ap.parse_args()

    if args.harness == "pi" and (args.model is None or args.pi_provider is None):
        prov_model = task_spec.PI_CANDIDATE_MODELS.get(args.candidate)
        if prov_model:
            args.pi_provider = args.pi_provider or prov_model[0]
            args.model = args.model or prov_model[1]
        if not args.model or not args.pi_provider:
            ap.error("pi harness needs --pi-provider and --model")
    elif args.model is None:
        args.model = task_spec.CANDIDATE_MODELS.get(args.candidate)
    if args.harness == "direct-fmapi" and (not args.model
                                           or not args.model.startswith("databricks-")):
        ap.error("direct-fmapi needs a databricks-* model (candidate default or --model)")

    tiers = list(task_spec.TIERS) if args.tier == "all" else [args.tier]
    for t in tiers:
        if t not in task_spec.TIERS:
            ap.error(f"unknown tier {t!r} (choose from {task_spec.TIERS} or 'all')")

    for t in tiers:
        meta = run_one_tier(args, t)
        if t == "tier1-gate" and args.tier == "all" and not meta["artifact_complete"]:
            print("[run_task] gate tier produced no complete app — stopping "
                  "(suite gate rule); grade tier1 to confirm.")
            break


if __name__ == "__main__":
    main()
