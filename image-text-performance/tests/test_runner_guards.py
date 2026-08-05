"""러너 안전장치 테스트 — 종료 코드·resume 구성 드리프트.

모델을 추가하거나 엔드포인트가 장애일 때 "조용히 성공한 것처럼" 끝나는 경로를 막는다.
실측 배경:
- 실패해도 항상 exit 0이라, 모델 전체가 403/400이어도 "실행 완료"가 찍히고 자동화가
  성공으로 오판했다.
- 모델을 추가한 뒤 `--resume`하면 기존 3모델 run에 새 모델 결과가 섞이는데 manifest에는
  옛 3모델만 남아, "이 리포트는 어떤 구성으로 뽑혔나"가 거짓이 됐다.
"""

import json
import os

import pytest

from src import runner
from src.results import RunManifest
from src.runner import GLOBAL_FAILURE_EXIT_RATE, _exit_code, _manifest_drift


def _cell(n=30, *, failed=0, unreliable=False, error=None):
    metrics = {"error": error} if error else {"accuracy": 0.9}
    e = {"n": n, "metrics": metrics}
    if failed:
        e["n_call_failed"] = failed
        e["call_failure_rate"] = failed / n
    if unreliable:
        e["unreliable"] = True
    return e


# ──────────────────────────────────────────────── 종료 코드


def test_clean_run_exits_zero():
    scores = {f"c{i}": _cell() for i in range(36)}
    assert _exit_code(scores, expected_cells=36) == 0


def test_scoring_error_exits_nonzero():
    """채점 예외 셀이 있으면 실패 — 코드·데이터 문제다."""
    assert _exit_code({"a": _cell(error="ValueError: boom")}, expected_cells=1) == 1


def test_unreliable_cell_exits_nonzero():
    """실패율 과다 셀은 점수가 장애의 산물이므로 실패로 본다."""
    assert _exit_code({"a": _cell(failed=19, unreliable=True)}, expected_cells=1) == 1


def test_total_failure_exits_nonzero():
    """모델 전체가 403/400이어도 exit 0이던 문제 — 전체 실패율이 임계 초과면 실패."""
    scores = {f"c{i}": _cell(failed=30) for i in range(36)}
    assert _exit_code(scores, expected_cells=36) == 1


def test_sporadic_failure_below_threshold_exits_zero():
    """산발적 소수 실패는 0 유지 — 그 셀은 채점에서 제외되고 리포트에 드러난다."""
    scores = {f"c{i}": _cell() for i in range(35)}
    scores["c35"] = _cell(failed=1)          # 1/1080 ≈ 0.1%
    assert _exit_code(scores, expected_cells=36) == 0


def test_failure_rate_threshold_boundary():
    """임계값(GLOBAL_FAILURE_EXIT_RATE)을 넘어야 실패 — 경계에서 흔들리지 않는다.

    실패를 여러 셀에 흩어 놓는다: 한 셀에 몰리면 그 셀이 unreliable이 돼 다른 조건으로
    실패하므로 '전체 실패율' 조건만 따로 확인할 수 없다.
    """
    n_cells, n = 10, 30
    assert GLOBAL_FAILURE_EXIT_RATE == pytest.approx(0.10), "임계값이 바뀌면 아래 수치도 조정"

    over = {f"c{i}": _cell(n, failed=4) for i in range(n_cells)}     # 40/300 = 13.3% > 10%
    assert _exit_code(over, expected_cells=n_cells) == 1

    under = {f"c{i}": _cell(n, failed=2) for i in range(n_cells)}    # 20/300 = 6.7% < 10%
    assert _exit_code(under, expected_cells=n_cells) == 0


def test_missing_cells_exits_nonzero():
    """기대 매트릭스보다 완료 셀이 적으면 실패(중간에 죽었거나 로드 실패)."""
    assert _exit_code({"a": _cell()}, expected_cells=36) == 1


# ──────────────────────────────────────────────── resume 구성 드리프트


def _manifest(models, *, tasks=("IMG-1",), modes=("minimal",), samples=30, profile="ai_devtools"):
    return RunManifest(
        run_id="r", created_at="t",
        models=[{"id": m} for m in models],
        reasoning_modes=list(modes), task_ids=list(tasks),
        git_commit="abc", samples_per_task=samples, profile=profile,
    )


def test_same_config_resume_allowed():
    """구성이 같으면 resume 허용 — 실패 셀만 재실행하는 정상 워크플로를 막지 않는다."""
    cur = _manifest(["opus", "sol", "glm"])
    prev = {"models": [{"id": "opus"}, {"id": "sol"}, {"id": "glm"}],
            "reasoning_modes": ["minimal"], "task_ids": ["IMG-1"],
            "samples_per_task": 30, "profile": "ai_devtools"}
    assert _manifest_drift(prev, cur) == []


def test_added_model_blocks_resume():
    """모델 추가 후 resume은 차단 — 서로 다른 구성이 한 run에 섞인다."""
    cur = _manifest(["opus", "sol", "glm", "sonnet"])
    prev = {"models": [{"id": "opus"}, {"id": "sol"}, {"id": "glm"}],
            "reasoning_modes": ["minimal"], "task_ids": ["IMG-1"],
            "samples_per_task": 30, "profile": "ai_devtools"}
    drift = _manifest_drift(prev, cur)
    assert drift and "sonnet" in drift[0]


def test_sample_count_change_blocks_resume():
    cur = _manifest(["opus"], samples=30)
    prev = {"models": [{"id": "opus"}], "reasoning_modes": ["minimal"],
            "task_ids": ["IMG-1"], "samples_per_task": 10, "profile": "ai_devtools"}
    assert any("샘플 수" in d for d in _manifest_drift(prev, cur))


def test_profile_change_blocks_resume():
    """프로파일이 다르면 워크스페이스가 달라 비교 전제가 바뀐다."""
    cur = _manifest(["opus"], profile="ai_devtools")
    prev = {"models": [{"id": "opus"}], "reasoning_modes": ["minimal"],
            "task_ids": ["IMG-1"], "samples_per_task": 30, "profile": "tokyo"}
    assert any("프로파일" in d for d in _manifest_drift(prev, cur))


def test_reasoning_mode_change_blocks_resume():
    cur = _manifest(["opus"], modes=("minimal", "full"))
    prev = {"models": [{"id": "opus"}], "reasoning_modes": ["minimal"],
            "task_ids": ["IMG-1"], "samples_per_task": 30, "profile": "ai_devtools"}
    assert any("reasoning" in d for d in _manifest_drift(prev, cur))


def test_empty_prev_manifest_is_permissive():
    """옛 run에 없던 필드는 비교하지 않는다(하위호환) — 빈 manifest는 통과."""
    assert _manifest_drift({}, _manifest(["opus"])) == []


# ──────────────────────────────────────────────── 완주 판정(2026-08-06 지적)


def test_load_failure_makes_run_fail():
    """태스크 로드 실패로 셀이 누락되면 exit 1 — 예전엔 기대 셀 수에서 빼서 exit 0이었다."""
    scores = {f"c{i}": _cell() for i in range(33)}
    assert _exit_code(scores, expected_cells=36,
                      failed_tasks={"TXT-2": "샘플 로드 실패: ValueError"}) == 1


def test_sample_shortfall_makes_run_fail():
    """요청보다 적게 채점되면 exit 1 — `--samples 30`은 완주 조건이다."""
    scores = {f"c{i}": _cell() for i in range(36)}
    assert _exit_code(scores, expected_cells=36, shortfall_tasks={"TXT-2": (10, 30)}) == 1


def test_judge_mass_failure_makes_run_fail():
    """judge가 대량 실패하면 exit 1 — 생성 태스크 대표 수치가 사실상 비어 있다."""
    scores = {
        f"c{i}": {"n": 30, "metrics": {"accuracy": 0.9,
                                       "judge_detail": {"n": 0, "n_failed": 30}}}
        for i in range(3)
    }
    assert _exit_code(scores, expected_cells=3) == 1


def test_judge_sporadic_failure_keeps_zero():
    """judge 산발 실패(임계 미만)는 정상 — 요약 태스크에서 30건당 1~2건은 흔하다."""
    scores = {
        f"c{i}": {"n": 30, "metrics": {"accuracy": 0.9,
                                       "judge_detail": {"n": 29, "n_failed": 1}}}
        for i in range(3)
    }
    assert _exit_code(scores, expected_cells=3) == 0


# ──────────────────────────────────────────────── run-id 충돌 방지


def test_run_id_avoids_existing_dir(tmp_path):
    """같은 분에 두 번 실행해도 run-id가 겹치지 않는다(결과가 섞이면 수치가 뒤엉킨다).

    make_run_id는 디렉터리 생성으로 **예약까지** 하므로, 호출부가 mkdir을 따로 하지 않는다.
    """
    from src.results import make_run_id

    res, rep = tmp_path / "results", tmp_path / "reports"
    first = make_run_id(results_root=res, reports_root=rep)
    assert (res / first).is_dir(), "예약 시 디렉터리를 만들어야 한다"
    second = make_run_id(results_root=res, reports_root=rep)
    assert second != first
    assert second.startswith(first)      # 같은 타임스탬프 + 접미사
    assert (res / second).is_dir()


def test_run_id_reserves_reports_dir_too(tmp_path):
    """reports/도 함께 예약한다 — results만 예약하면 `--out`이 다른 두 프로세스가 같은
    reports/<id>/를 덮어쓴다(리포트 경로는 항상 프로세스 기준 reports/)."""
    from src.results import make_run_id

    res, rep = tmp_path / "results", tmp_path / "reports"
    rid = make_run_id(results_root=res, reports_root=rep)
    assert (rep / rid).is_dir(), "reports 쪽도 예약해야 덮어쓰기를 막는다"

    # reports에만 같은 이름이 이미 있으면 그 ID는 건너뛴다(옛 리포트 보호).
    other_res = tmp_path / "results2"
    rid2 = make_run_id(results_root=other_res, reports_root=rep)
    assert rid2 != rid, "reports가 점유된 ID를 재사용하면 옛 리포트를 덮어쓴다"
    assert not (other_res / rid).exists(), "포기한 후보의 results 디렉터리는 되돌려야 한다"


def test_run_id_reservation_is_atomic_under_concurrency(tmp_path):
    """동시 실행에서도 유일한 ID를 준다 — exists()+mkdir(exist_ok=True)의 TOCTOU 방지."""
    from concurrent.futures import ThreadPoolExecutor

    from src.results import make_run_id

    res, rep = tmp_path / "results", tmp_path / "reports"
    with ThreadPoolExecutor(max_workers=12) as ex:
        ids = list(ex.map(
            lambda _: make_run_id(results_root=res, reports_root=rep), range(12)
        ))

    assert len(set(ids)) == len(ids), f"동시 예약에서 중복 발생: {sorted(ids)}"


def test_run_id_plain_when_no_conflict(tmp_path):
    """충돌이 없으면 접미사 없이 기존 형식(YYYY-MM-DDTHH-MM)을 유지한다."""
    import re

    from src.results import make_run_id

    rid = make_run_id(
        results_root=tmp_path / "results", reports_root=tmp_path / "reports"
    )
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}", rid), rid


# ──────────────────────────────────────────────── --fresh는 삭제가 아니라 보관 이동


def test_purge_moves_instead_of_deleting(tmp_path, monkeypatch):
    """--fresh가 이전 run을 .trash로 **이동**한다 — 새 run이 죽어도 복구 가능해야 한다."""
    from src.runner import _purge_previous_runs

    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    for base in (results, reports):
        (base / "old-run").mkdir(parents=True)
        (base / "old-run" / "f.txt").write_text("data", encoding="utf-8")
    (results / "keep-me").mkdir()

    _purge_previous_runs(results, reports, keep_run_id="keep-me")

    assert not (results / "old-run").exists(), "이전 run이 남아 있다"
    assert (results / "keep-me").exists(), "이번 run 디렉터리를 지웠다"
    moved = list((tmp_path / ".trash").glob("purged-*/results/old-run/f.txt"))
    assert moved, "삭제만 하고 보관하지 않았다 — 사고 시 복구할 수 없다"
    assert moved[0].read_text(encoding="utf-8") == "data"


# ──────────────────────────── judge 미실행/예외 (P0-2, 2026-08-06)


def test_judge_error_only_cell_fails():
    """judge_scores 예외로 judge_error만 남은 셀은 완주가 아니다(실측 재현: rc=0이었다)."""
    scores = {"a": {"n": 30, "metrics": {"accuracy": 0.9, "judge_error": "FMAPIError: 403"}}}
    assert _exit_code(scores, expected_cells=1, judge_expected_cells={"a"}) == 1


def test_judge_expected_but_missing_fails():
    """judge 대상 셀에 judge_mean/judge_detail이 없으면 실패."""
    scores = {"a": {"n": 30, "metrics": {"accuracy": 0.9}}}
    assert _exit_code(scores, expected_cells=1, judge_expected_cells={"a"}) == 1


def test_judge_client_init_failure_fails():
    """judge 클라이언트 초기화 실패를 조용히 넘기지 않는다(--no-judge를 쓰라는 신호)."""
    scores = {"a": {"n": 30, "metrics": {"accuracy": 0.9}}}
    assert _exit_code(scores, expected_cells=1, judge_init_error="FMAPIError: auth") == 1


def test_judge_present_passes():
    """judge가 정상 채점됐으면 통과."""
    scores = {
        "a": {"n": 30, "metrics": {"accuracy": 0.9, "judge_mean": 4.5,
                                   "judge_detail": {"n": 30}}}
    }
    assert _exit_code(scores, expected_cells=1, judge_expected_cells={"a"}) == 0


def test_no_judge_run_passes_without_judge_metrics():
    """--no-judge 실행은 judge 대상 셀이 없으므로 judge 수치가 없어도 통과."""
    scores = {"a": {"n": 30, "metrics": {"accuracy": 0.9}}}
    assert _exit_code(scores, expected_cells=1, judge_expected_cells=set()) == 0


# ──────────────────────────── 부분 셀 이어달리기 (P1-3, 2026-08-06)


def _sample_row(model, task, sid, *, error=False):
    import json as _json

    return _json.dumps({
        "model_id": model, "task_id": task, "sample_id": sid,
        "reasoning_mode": "minimal", "prompt": "p",
        "model_output": "__ERROR__: boom" if error else "ok",
        "reference": "x", "request_id": None,
        "finish_reason": "error" if error else "stop",
        "usage": {}, "latency_ms_local": 1.0, "timestamp": "t",
    }, ensure_ascii=False)


def test_partial_progress_preserves_partial_rows(tmp_path):
    """미완료 셀의 성공 행을 **보존**하고 이어달리기 지점을 알려준다.

    예전엔 미완료 셀 행을 전부 버려서, 5샘플마다 flush해도 중단된 셀은 30개를 다시
    호출했다(SAMPLE_FLUSH_EVERY가 복구에 도움이 안 됐다 — 2026-08-06 지적).
    """
    import json as _json

    from src.runner import _partial_progress

    sp = tmp_path / "samples.jsonl"
    lines = [_sample_row("opus", "TXT-1", i) for i in range(2)]          # 완료 셀
    lines += [_sample_row("opus", "TXT-2", i) for i in range(7)]         # 부분 완료
    lines += [_sample_row("opus", "TXT-2", 7, error=True)]               # 실패 → 재시도 대상
    sp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    progress = _partial_progress(sp, {"opus::TXT-1::minimal"})

    assert progress == {"opus::TXT-2::minimal": set(range(7))}
    kept = [_json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(kept) == 9, f"완료 2 + 부분 7이 보존돼야 한다(실제 {len(kept)})"
    assert all(r["finish_reason"] != "error" for r in kept), "실패 행이 남았다"


def test_partial_progress_drops_poisoned_cell_rows(tmp_path):
    """오염 셀(>50% 실패)의 행은 재사용하지 않는다 — 그 응답들은 재시도 대상이다."""
    from src.runner import _partial_progress

    sp = tmp_path / "samples.jsonl"
    sp.write_text(
        "\n".join(_sample_row("sol", "IMG-2", i, error=True) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    progress = _partial_progress(sp, set(), drop_keys={"sol::IMG-2::minimal"})

    assert progress == {}
    assert sp.read_text(encoding="utf-8").strip() == "", "오염 셀 행이 남았다"


def test_partial_progress_no_file_is_noop(tmp_path):
    from src.runner import _partial_progress

    assert _partial_progress(tmp_path / "missing.jsonl", set()) == {}


# ──────────────────────────── 필수 산출물 실패 (P1-5, review 지적)


def test_artifact_errors_make_run_fail():
    """리포트·인덱스·차트 생성 실패는 exit 1 — "성공했는데 리포트가 없는" 상태 방지.

    (gstack review 지적) `artifact_errors`를 _exit_code에 넣었지만 테스트가 없었다.
    """
    scores = {f"c{i}": _cell() for i in range(36)}
    assert _exit_code(scores, expected_cells=36,
                      artifact_errors=["report.md: OSError: disk full"]) == 1
    assert _exit_code(scores, expected_cells=36, artifact_errors=["chart_*.png 없음"]) == 1


def test_empty_artifact_errors_passes():
    """산출물이 모두 정상이면 통과(과잉 실패 방지)."""
    scores = {f"c{i}": _cell() for i in range(36)}
    assert _exit_code(scores, expected_cells=36, artifact_errors=[]) == 0
    assert _exit_code(scores, expected_cells=36, artifact_errors=None) == 0


# ──────────────────────────── 부분 재사용 행 로드 (성능 지적)


def test_load_cell_rows_streams_only_matching_cell(tmp_path):
    """지정한 셀·sample_id 행만 돌려준다 — 파일 전체를 메모리에 올리지 않는다.

    (gstack review 지적) 예전엔 셀 루프 안에서 load_sample_results()로 파일 전체를
    리스트로 만들어, 셀마다 전체 재스캔(O(셀수 × 행수)) + O(1) 메모리 계약 위반이었다.
    """
    from src.runner import _load_cell_rows

    sp = tmp_path / "samples.jsonl"
    lines = [_sample_row("opus", "TXT-2", i) for i in range(5)]
    lines += [_sample_row("sol", "TXT-2", i) for i in range(5)]     # 다른 모델
    lines += [_sample_row("opus", "TXT-9", i) for i in range(5)]    # 다른 태스크
    sp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = _load_cell_rows(sp, "opus", "TXT-2", "minimal", {0, 2, 4})

    assert set(rows) == {0, 2, 4}, f"다른 셀 행이 섞였거나 누락됐다: {sorted(rows)}"
    assert all(r.model_id == "opus" and r.task_id == "TXT-2" for r in rows.values())


def test_load_cell_rows_handles_missing_file_and_empty_ids(tmp_path):
    from src.runner import _load_cell_rows

    assert _load_cell_rows(tmp_path / "nope.jsonl", "opus", "T", "minimal", {1}) == {}
    sp = tmp_path / "samples.jsonl"
    sp.write_text(_sample_row("opus", "T", 0) + "\n", encoding="utf-8")
    assert _load_cell_rows(sp, "opus", "T", "minimal", set()) == {}


def test_load_cell_rows_skips_corrupt_lines(tmp_path):
    """손상된 JSON 행은 건너뛴다(그 샘플은 재호출된다) — 셀 전체를 죽이지 않는다."""
    from src.runner import _load_cell_rows

    sp = tmp_path / "samples.jsonl"
    sp.write_text(
        _sample_row("opus", "T", 0) + "\n{not json}\n" + _sample_row("opus", "T", 1) + "\n",
        encoding="utf-8",
    )
    rows = _load_cell_rows(sp, "opus", "T", "minimal", {0, 1})
    assert set(rows) == {0, 1}


# ---------------------------------------------------------------------------
# tmp 파일 누수 / 재작성 실패 시 원본 보존
# (adversarial 리뷰 지적: os.replace 실패 시 .tmp가 영구히 남고, _partial_progress가
#  예외를 던지면 run 전체가 죽는다. 셀마다 호출되므로 누수는 계속 쌓인다.)
# ---------------------------------------------------------------------------

def test_atomic_write_json_leaves_no_tmp_on_failure(tmp_path, monkeypatch):
    """os.replace가 실패해도 .tmp를 남기지 않고, 원본은 그대로 유지된다."""
    target = tmp_path / "scores.json"
    target.write_text('{"old": true}', encoding="utf-8")

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        runner._atomic_write_json(target, {"new": True})
    monkeypatch.setattr(os, "replace", real_replace)

    assert not (tmp_path / "scores.json.tmp").exists(), ".tmp가 남으면 셀마다 쌓인다"
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}, "원본이 보존돼야 함"


def test_partial_progress_rewrite_failure_preserves_original(tmp_path, monkeypatch, capsys):
    """재작성이 실패하면 이어달리기를 포기하되(빈 dict), 원본 samples.jsonl은 살아 있다."""
    samples = tmp_path / "samples.jsonl"
    rows = [
        {"model_id": "opus", "task_id": "IMG-1", "reasoning_mode": "minimal",
         "sample_id": 0, "finish_reason": "stop"},
        {"model_id": "opus", "task_id": "IMG-1", "reasoning_mode": "minimal",
         "sample_id": 1, "finish_reason": "stop"},
    ]
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    before = samples.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("read-only file system")

    monkeypatch.setattr(os, "replace", boom)
    progress = runner._partial_progress(samples, set(), drop_keys=set())

    assert progress == {}, "재작성 실패 시 재사용을 포기해야 한다(반쪽 파일로 채점하면 오염)"
    assert not (tmp_path / "samples.jsonl.tmp").exists(), ".tmp를 남기지 않아야 함"
    assert samples.read_text(encoding="utf-8") == before, "원본이 그대로 보존돼야 함"
    assert "재작성 실패" in capsys.readouterr().out, "조용히 넘어가면 안 됨"


# ---------------------------------------------------------------------------
# 완주 판정: 개수 비교가 아니라 키 집합 비교
# (codex 지적) 옛 run에서 남은 scores 항목이 누락 셀을 벌충해 exit 0이 되고,
# 그 낡은 항목이 리포트 순위에도 들어간다.
# ---------------------------------------------------------------------------

def test_stale_score_entry_cannot_mask_missing_cell():
    """개수는 맞지만 키가 다르면 실패로 판정한다(옛 결과가 섞인 상태)."""
    expected = {"opus::IMG-1::minimal", "sol::IMG-1::minimal"}
    scores = {
        "opus::IMG-1::minimal": _cell(),
        "glm::TXT-9::minimal": _cell(),   # 이번 매트릭스에 없는 옛 항목
    }
    assert _exit_code(scores, expected_cells=2, expected_keys=expected) == 1


def test_exact_key_match_passes():
    expected = {"opus::IMG-1::minimal", "sol::IMG-1::minimal"}
    scores = {k: _cell() for k in expected}
    assert _exit_code(scores, expected_cells=2, expected_keys=expected) == 0


# ---------------------------------------------------------------------------
# samples.jsonl 중복 행: 마지막 것만 집계 (재호출 후 append로 생기는 중복)
# ---------------------------------------------------------------------------

def test_duplicate_sample_rows_counted_once(tmp_path):
    """같은 (모델·태스크·모드·샘플) 행이 두 번 있으면 마지막 것만 남는다.

    중복을 그대로 세면 시간·비용·토큰·실패율이 그 샘플만큼 부풀려진다.
    """
    from src.results import load_sample_results

    def row(out, latency):
        return {
            "model_id": "opus", "task_id": "IMG-1", "sample_id": 3,
            "reasoning_mode": "minimal", "prompt": "p", "model_output": out,
            "reference": "r", "request_id": None, "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "latency_ms_local": latency, "timestamp": "2026-08-06T00:00:00Z",
        }

    (tmp_path / "samples.jsonl").write_text(
        json.dumps(row("옛 응답", 1000.0)) + "\n" + json.dumps(row("새 응답", 2000.0)) + "\n",
        encoding="utf-8",
    )
    rows = load_sample_results(tmp_path)
    assert len(rows) == 1, "중복 행이 두 번 집계되면 비용·시간이 부풀려진다"
    assert rows[0].model_output == "새 응답", "실제 채점에 쓰인 마지막 응답이어야 한다"
    assert rows[0].latency_ms_local == 2000.0


# ---------------------------------------------------------------------------
# .trash 보관: 옛 보관본은 이번 보관본이 완성된 뒤에 지운다
# ---------------------------------------------------------------------------

def test_purge_keeps_old_backup_until_new_one_is_complete(tmp_path, monkeypatch, capsys):
    """이동이 중간에 실패하면 옛 보관본이 살아 있어야 한다.

    옛 순서(옛것 삭제 → 이동)에서는 실패 시 옛 보관본도 없고 이번 보관본도 반쪽이 됐다.
    """
    import shutil

    monkeypatch.chdir(tmp_path)
    results, reports = tmp_path / "results", tmp_path / "reports"
    (results / "old-run").mkdir(parents=True)
    (results / "keep-me").mkdir()
    reports.mkdir()
    prior = tmp_path / ".trash" / "purged-19990101-000000"
    prior.mkdir(parents=True)
    (prior / "marker.txt").write_text("직전 보관본", encoding="utf-8")

    def boom(src, dst):
        raise OSError("cross-device link failed")

    monkeypatch.setattr(shutil, "move", boom)
    with pytest.raises(OSError):
        runner._purge_previous_runs(results, reports, keep_run_id="keep-me")

    assert (prior / "marker.txt").exists(), "이동 실패 시 직전 보관본이 남아 있어야 한다"
    assert (results / "old-run").exists(), "이동에 실패했으면 원본도 그대로여야 한다"


def test_purge_removes_old_backup_after_success(tmp_path, monkeypatch):
    """정상 이동 후에는 옛 보관본을 정리해 디스크가 무한히 늘지 않는다."""
    monkeypatch.chdir(tmp_path)
    results, reports = tmp_path / "results", tmp_path / "reports"
    (results / "old-run").mkdir(parents=True)
    (results / "keep-me").mkdir()
    reports.mkdir()
    prior = tmp_path / ".trash" / "purged-19990101-000000"
    prior.mkdir(parents=True)

    runner._purge_previous_runs(results, reports, keep_run_id="keep-me")

    assert not prior.exists(), "성공 후에는 직전 보관본을 정리한다"
    assert not (results / "old-run").exists(), "옛 run은 이동됐어야 한다"
    assert (results / "keep-me").exists(), "이번 run은 건드리지 않는다"
    current = list((tmp_path / ".trash").glob("purged-*"))
    assert len(current) == 1
    assert (current[0] / "results" / "old-run").is_dir(), "보관본에 옛 run이 들어 있어야 한다"
