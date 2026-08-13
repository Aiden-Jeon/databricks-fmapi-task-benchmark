#!/usr/bin/env python
"""K-MLE-Bench job runner: executes one (lane × task) agent attempt on a cluster.

Usage (spark_python_task parameters): runner.py <LANE> <TASK> [smoke|full]
  LANE ∈ L1 (Claude Code + databricks-claude-opus-5)
         L2 (Codex CLI  + gpt-5.6-sol via codex route)
         L3 (opencode   + databricks-glm-5-2 via mlflow route)

Env (from spark_env_vars): KMLE_WS (workspace URL), KMLE_PAT (secret).
Reads pack from   /Volumes/newjeans_ontos_catalog/kmle_packs/packs/<TASK>
Writes artifacts  /Volumes/newjeans_ontos_catalog/kmle_results/artifacts/<RUN_ID>/
The private answers volume is never touched here.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

LANE, TASK = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "full"
CATALOG_ARG = sys.argv[4] if len(sys.argv) > 4 else "newjeans_ontos_catalog"

# Serverless-only workspace: no spark_env_vars. Resolve host from ambient SDK
# auth and the PAT from the kmle secret scope; env vars remain as local override.
WS = os.environ.get("KMLE_WS", "")
PAT = os.environ.get("KMLE_PAT", "")
if not (WS and PAT):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.runtime import dbutils
    WS = WS or WorkspaceClient().config.host
    PAT = PAT or dbutils.secrets.get("kmle", "pat")
WS = WS.rstrip("/")

CATALOG = CATALOG_ARG
PACKS = f"/Volumes/{CATALOG}/kmle_packs/packs"
ARTIFACTS = f"/Volumes/{CATALOG}/kmle_results/artifacts"
RUN_ID = f"{LANE}_{TASK}_{MODE}_{time.strftime('%Y%m%d_%H%M%S')}"

WORK = pathlib.Path("/tmp/kmle") / RUN_ID
HOME2 = WORK / "home"
TASKDIR = WORK / "task"
NODE_DIR = pathlib.Path("/tmp/kmle_node")
NPM_PREFIX = pathlib.Path("/tmp/kmle_npm")
# Serverless fleet mixes x86_64 and aarch64 hosts — pick the matching build.
import platform
_ARCH = "arm64" if platform.machine() in ("aarch64", "arm64") else "x64"
NODE_URL = f"https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-{_ARCH}.tar.gz"

TIMEOUT = 600 if MODE == "smoke" else 7200  # 10 min smoke, 2 h full

# M-track: fixed harness (opencode, pinned) x swapped model. The opencode.json
# written for M1/M2/M3 is byte-identical; only the -m selector differs.
OPENCODE_PIN = "opencode-ai@0.0.0-beta-202605152242"
M_SELECTORS = {"M1": "databricks-anthropic/databricks-claude-opus-5",
               "M2": "databricks-oss/databricks-gpt-5-6-sol",
               "M3": "databricks-oss/databricks-glm-5-2",
               "M4": "databricks-oss/databricks-qwen35-122b-a10b",
               "M5": "databricks-oss/databricks-llama-4-maverick",
               "M6": "databricks-oss/databricks-gpt-oss-120b",
               "M7": "databricks-oss/databricks-kimi-k3"}
OSS_LIMIT = {"limit": {"context": 128000, "output": 16000}}


def sh(cmd, **kw):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def ensure_toolchain():
    if not (NODE_DIR / "bin" / "node").exists():
        sh(f"mkdir -p {NODE_DIR} && curl -fsSL {NODE_URL} | tar -xz -C {NODE_DIR} --strip-components=1")
    os.environ["PATH"] = f"{NODE_DIR}/bin:{NPM_PREFIX}/bin:" + os.environ["PATH"]
    os.environ["NPM_CONFIG_PREFIX"] = str(NPM_PREFIX)
    pkg = ({"L1": "@anthropic-ai/claude-code", "L2": "@openai/codex",
            "L3": "opencode-ai"}.get(LANE)
           or (OPENCODE_PIN if LANE in M_SELECTORS else None))
    binname = "opencode" if LANE in M_SELECTORS else \
        {"L1": "claude", "L2": "codex", "L3": "opencode"}[LANE]
    if pkg is None:
        raise SystemExit(f"unknown lane {LANE}")
    if not (NPM_PREFIX / "bin" / binname).exists():
        sh(f"npm install -g --silent {pkg}")


def lane_env_and_cmd(prompt_file: pathlib.Path):
    env = dict(os.environ, HOME=str(HOME2))
    prompt_arg = f'"$(cat {prompt_file})"'
    if LANE == "L1":
        env.update({
            "ANTHROPIC_BASE_URL": f"{WS}/ai-gateway/anthropic",
            "ANTHROPIC_AUTH_TOKEN": PAT,
            "ANTHROPIC_CUSTOM_HEADERS": "x-databricks-use-coding-agent-mode: true",
            "CLAUDE_CODE_USE_GATEWAY": "1",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "databricks-claude-opus-5",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",  # ephemeral job container; allows headless flags as root
        })
        cmd = (f"claude -p {prompt_arg} --model databricks-claude-opus-5 "
               f"--output-format json --dangerously-skip-permissions "
               f"--disallowedTools WebSearch,WebFetch")
    elif LANE == "L2":
        cfg = HOME2 / ".codex"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "config.toml").write_text(
            'model = "gpt-5.6-sol"\nmodel_provider = "databricks"\n\n'
            "[model_providers.databricks]\n"
            'name = "Databricks AI Gateway"\n'
            f'base_url = "{WS}/ai-gateway/codex/v1"\n'
            'wire_api = "responses"\n'
            'env_key = "DATABRICKS_TOKEN"\n')
        env["DATABRICKS_TOKEN"] = PAT
        cmd = (f"codex exec --skip-git-repo-check --sandbox danger-full-access "
               f"{prompt_arg}")
    elif LANE == "L3":
        cfg = HOME2 / ".config" / "opencode"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "opencode.json").write_text(json.dumps({
            "model": "databricks-oss/databricks-glm-5-2",
            "provider": {"databricks-oss": {
                "npm": "@ai-sdk/openai",
                "options": {"baseURL": f"{WS}/ai-gateway/mlflow/v1",
                            "apiKey": PAT,
                            "headers": {"Authorization": f"Bearer {PAT}"}},
                "models": {"databricks-glm-5-2": {}}}},
            "permission": {"edit": "allow", "bash": "allow"},
        }))
        cmd = (f"opencode run -m databricks-oss/databricks-glm-5-2 {prompt_arg}")
    elif LANE in M_SELECTORS:
        cfg = HOME2 / ".config" / "opencode"
        cfg.mkdir(parents=True, exist_ok=True)
        auth = {"Authorization": f"Bearer {PAT}"}
        (cfg / "opencode.json").write_text(json.dumps({
            "provider": {
                "databricks-anthropic": {
                    "npm": "@ai-sdk/anthropic",
                    "options": {"baseURL": f"{WS}/ai-gateway/anthropic/v1",
                                "apiKey": PAT, "headers": auth},
                    "models": {"databricks-claude-opus-5":
                               {"options": {"toolStreaming": False}}}},
                "databricks-oss": {
                    "npm": "@ai-sdk/openai",
                    "options": {"baseURL": f"{WS}/ai-gateway/mlflow/v1",
                                "apiKey": PAT, "headers": auth},
                    "models": {"databricks-gpt-5-6-sol": {},
                               "databricks-glm-5-2": {},
                               "databricks-qwen35-122b-a10b": OSS_LIMIT,
                               "databricks-llama-4-maverick": OSS_LIMIT,
                               "databricks-gpt-oss-120b": OSS_LIMIT,
                               "databricks-kimi-k3": OSS_LIMIT}}},
            "permission": {"edit": "allow", "bash": "allow"},
        }, sort_keys=True))
        cmd = f"opencode run -m {M_SELECTORS[LANE]} {prompt_arg}"
    else:
        raise SystemExit(f"unknown lane {LANE}")
    return env, cmd


def main():
    t0 = time.time()
    HOME2.mkdir(parents=True, exist_ok=True)
    (TASKDIR / "outputs").mkdir(parents=True, exist_ok=True)

    if MODE == "smoke":
        prompt = "Create a file named outputs/ready.txt containing exactly: S1-JOB-OK"
    else:
        shutil.copytree(f"{PACKS}/{TASK}", TASKDIR, dirs_exist_ok=True)
        kickoff = pathlib.Path(f"{PACKS}/_harness/kickoff_prompt_ko.md").read_text()
        prompt = kickoff

    ensure_toolchain()
    pf = WORK / "prompt.txt"
    pf.write_text(prompt)
    env, cmd = lane_env_and_cmd(pf)

    result = {"run_id": RUN_ID, "lane": LANE, "task": TASK, "mode": MODE,
              "arch": platform.machine(), "cpus": os.cpu_count(),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log = WORK / "agent_stdout.log"
    try:
        with open(log, "w") as lf:
            p = subprocess.run(cmd, shell=True, cwd=TASKDIR, env=env,
                               stdin=subprocess.DEVNULL,  # codex exec blocks on open stdin
                               stdout=lf, stderr=subprocess.STDOUT,
                               timeout=TIMEOUT)
        result["exit_code"] = p.returncode
        result["timed_out"] = False
    except subprocess.TimeoutExpired:
        result["exit_code"] = -1
        result["timed_out"] = True
    result["wall_seconds"] = round(time.time() - t0, 1)
    sub = TASKDIR / "outputs" / ("ready.txt" if MODE == "smoke" else "submission.csv")
    result["artifact_present"] = sub.exists()

    out = pathlib.Path(f"{ARTIFACTS}/{RUN_ID}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    shutil.copy(log, out / "agent_stdout.log")
    for rel in ["outputs", "solution"]:
        src = TASKDIR / rel
        if src.exists():
            shutil.copytree(src, out / rel, dirs_exist_ok=True)

    print(json.dumps(result, indent=2), flush=True)
    if MODE == "smoke" and not result["artifact_present"]:
        raise SystemExit(f"SMOKE FAILED for {LANE}")


if __name__ == "__main__":
    main()
