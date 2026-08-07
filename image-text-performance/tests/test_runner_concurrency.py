"""병렬 호출이 채점 수치를 바꾸지 않음을 고정하는 통합 테스트.

속도 개선(셀 내 샘플 호출 병렬화)의 안전성은 단 하나의 성질로 요약된다:
**concurrency와 무관하게 scores.json이 동일하다.** 완료 순서가 뒤섞여도 결과가 샘플에
정확히 대응돼야 하며(map_concurrent 순서 보존 + reduce가 sample_id로 조회), 호출 실패
분류(CALL_FAILED→n_skipped)도 그대로 유지돼야 한다.

scores.json에는 벽시계 지연이 들어가지 않으므로(그건 samples.jsonl), 결정적 가짜 응답에
대해 c=1과 c=8의 scores.json은 **완전히 같아야** 한다. 다르면 병렬화가 수치를 오염시킨 것이다.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from src import runner
from src.adapters.fmapi import build_text_message
from src.config import load_models_config
from src.scoring.accumulators import MultiMeanAccumulator
from src.tasks.base import Sample, Task


# ── 결정적 가짜 태스크 ────────────────────────────────────────────────────────
# 프롬프트에 sample_id를 실어 보내고, 가짜 모델이 "got-<id>"로 답한다. 채점은 응답이
# 그 샘플의 기대값과 맞는지 본다 → 병렬 실행이 응답을 엉뚱한 샘플에 붙이면 점수가 틀어진다.
class _FakeTask(Task):
    task_id = "TXT-FAKE"
    kind = "qa"
    is_vision = False

    def load_samples(self, n, seed):
        return [
            Sample(sample_id=i, inputs={"q": f"question-{i}"}, reference=[f"ans-{i}"], lang="en")
            for i in range(n)
        ]

    def build_prompt(self, sample):
        # sample_id를 프롬프트에 실어 가짜 모델이 어떤 샘플인지 알게 한다.
        return build_text_message(f"SAMPLE {sample.sample_id}: {sample.inputs['q']}")

    def parse_output(self, raw_text, sample):
        return raw_text.strip()

    def make_accumulator(self):
        # 응답이 이 샘플의 기대값("got-<id>")과 정확히 일치하면 1.0.
        def _correct(pred, sample):
            return 1.0 if pred == f"got-{sample.sample_id}" else 0.0

        return MultiMeanAccumulator({"correct": _correct})


class _FakeResp:
    def __init__(self, text, finish="stop"):
        self.text = text
        self.request_id = "req"
        self.finish_reason = finish
        self.usage = {"prompt_tokens": 5, "completion_tokens": 3}


class _FakeFmapi:
    """가짜 FMAPI. 완료 순서를 뒤섞기 위해 무작위 지연을 넣는다.

    fail_ids에 든 sample_id는 예외를 던져 호출 실패(CALL_FAILED) 경로를 태운다.
    """

    def __init__(self, fail_ids=frozenset(), jitter=True):
        self.fail_ids = set(fail_ids)
        self.jitter = jitter

    def chat(self, endpoint, messages, *, max_tokens, extra_params=None,
             timeout_seconds=None, max_retries=None):
        text = messages[0]["content"]
        sid = int(text.split("SAMPLE ", 1)[1].split(":", 1)[0])
        if self.jitter:
            time.sleep(random.uniform(0, 0.02))   # 완료 순서를 제출 순서와 어긋나게
        if sid in self.fail_ids:
            raise RuntimeError(f"HTTP 502 upstream (s{sid})")
        return _FakeResp(f"got-{sid}")

    def close(self):
        pass


def _stub_reports(monkeypatch):
    """report/index 생성을 스텁으로 — 네트워크(judge Executive Summary)·실 파일 생성 회피.

    scores.json 비교만이 목적이므로 리포트 산출물은 필요한 파일만 만들어 둔다.
    """
    def fake_report(run_dir, results, scores, models_cfg, **kw):
        rd = Path("reports") / Path(run_dir).name
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "report.md").write_text("stub", encoding="utf-8")
        (rd / "facts.json").write_text("{}", encoding="utf-8")
        (rd / "presentation.html").write_text("<html></html>", encoding="utf-8")
        (rd / "chart_x.png").write_bytes(b"\x89PNG")
        return rd / "report.md"

    monkeypatch.setattr("src.report.generate.generate_report", fake_report)
    monkeypatch.setattr("src.report.index.rebuild_index", lambda **kw: Path("reports/index.md"))


# 리포지토리 루트(이 테스트 파일 기준) — chdir 후에도 config를 절대경로로 읽기 위함.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_once(tmp_path, monkeypatch, *, concurrency, fail_ids=frozenset(), n=12):
    """가짜 태스크·fmapi로 _run_samples를 1회 돌리고 scores.json을 돌려준다."""
    tmp_path.mkdir(parents=True, exist_ok=True)   # 호출부가 하위 경로를 넘길 수 있다
    # config를 먼저(chdir 전) 절대경로로 로드한다 — _run_samples는 리포트만 cwd 기준으로
    # 쓰므로, cwd를 tmp로 옮겨도 config는 repo에서 읽어야 한다.
    models_cfg = load_models_config(_REPO_ROOT / "config" / "models.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.tasks.loader.discover_tasks", lambda: {"TXT-FAKE": _FakeTask})
    # registry/reproducibility 메타는 repo 파일을 읽으므로 chdir 후엔 없다 — 가짜 태스크는
    # registry를 안 쓰므로 빈 dict로 스텁한다(_run_samples 상단 load_registry 호출 대비).
    monkeypatch.setattr("src.datasets_loader.load_registry", lambda *a, **k: {})
    _stub_reports(monkeypatch)
    model = models_cfg.get_model("sol")
    tasks_cfg = {
        "defaults": {"samples": n, "seed": 42},
        "image_tasks": [],
        "text_tasks": [{"id": "TXT-FAKE", "kind": "qa", "metrics": ["correct"]}],
    }
    matrix = [{
        "model_id": model.id, "model_endpoint": model.endpoint,
        "task_id": "TXT-FAKE", "reasoning_mode": "minimal",
    }]

    run_dir = tmp_path / "results" / f"run-c{concurrency}"
    run_dir.mkdir(parents=True)

    rc = runner._run_samples(
        fmapi=_FakeFmapi(fail_ids=fail_ids),
        models_cfg=models_cfg, tasks_cfg=tasks_cfg, matrix=matrix,
        run_dir=run_dir, sample_cap=n, enable_judge=False,
        results_root=tmp_path / "results", concurrency_override=concurrency,
    )
    scores = json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))
    return rc, scores


def test_scores_identical_across_concurrency(tmp_path, monkeypatch):
    """c=1과 c=8의 scores.json이 완전히 같다 — 병렬화가 수치를 바꾸지 않는다."""
    random.seed(0)
    _, seq = _run_once(tmp_path / "seq", monkeypatch, concurrency=1)
    random.seed(0)
    _, par = _run_once(tmp_path / "par", monkeypatch, concurrency=8)

    # run_dir 이름만 다를 뿐 셀 키·metrics는 동일해야 한다.
    assert seq.keys() == par.keys()
    for k in seq:
        assert seq[k]["metrics"] == par[k]["metrics"], f"{k}: 병렬 결과가 순차와 다르다"
        assert seq[k]["n"] == par[k]["n"]


def test_parallel_maps_responses_to_correct_samples(tmp_path, monkeypatch):
    """완료 순서가 뒤섞여도 응답이 올바른 샘플에 대응된다(정답률 1.0).

    reduce가 sample_id로 조회하므로 정렬이 깨지지 않는다. 미스매치가 있으면 correct<1.0.
    """
    random.seed(1)
    _, scores = _run_once(tmp_path, monkeypatch, concurrency=8, n=12)
    cell = scores["sol::TXT-FAKE::minimal"]["metrics"]
    assert cell["correct"] == 1.0, f"응답이 엉뚱한 샘플에 붙었다: {cell}"
    assert cell["n_evaluated"] == 12


def test_call_failures_classified_under_concurrency(tmp_path, monkeypatch):
    """병렬 실행에서도 호출 실패는 CALL_FAILED로 분류돼 분모에서 빠진다(n_skipped)."""
    random.seed(2)
    # 12개 중 3개 호출 실패 → 나머지 9개만 채점, 전부 정답이므로 correct=1.0.
    _, scores = _run_once(tmp_path, monkeypatch, concurrency=8, fail_ids={2, 5, 9}, n=12)
    cell = scores["sol::TXT-FAKE::minimal"]["metrics"]
    assert cell["n_evaluated"] == 9, f"호출 실패가 분모에 섞였다: {cell}"
    assert cell["n_skipped"] == 3, f"호출 실패 수가 안 맞는다: {cell}"
    assert cell["correct"] == 1.0
    # 셀 수준 호출 실패 카운트도 기록돼야 한다.
    assert scores["sol::TXT-FAKE::minimal"]["n_call_failed"] == 3


def test_call_failures_identical_across_concurrency(tmp_path, monkeypatch):
    """실패가 섞인 셀도 c=1과 c=8의 scores.json이 같다."""
    random.seed(3)
    _, seq = _run_once(tmp_path / "seq", monkeypatch, concurrency=1, fail_ids={1, 7})
    random.seed(3)
    _, par = _run_once(tmp_path / "par", monkeypatch, concurrency=8, fail_ids={1, 7})
    for k in seq:
        assert seq[k]["metrics"] == par[k]["metrics"]
        assert seq[k].get("n_call_failed") == par[k].get("n_call_failed")
