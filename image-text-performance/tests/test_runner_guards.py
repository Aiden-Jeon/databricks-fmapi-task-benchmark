"""러너 안전장치 테스트 — 종료 코드·resume 구성 드리프트.

모델을 추가하거나 엔드포인트가 장애일 때 "조용히 성공한 것처럼" 끝나는 경로를 막는다.
실측 배경:
- 실패해도 항상 exit 0이라, 모델 전체가 403/400이어도 "실행 완료"가 찍히고 자동화가
  성공으로 오판했다.
- 모델을 추가한 뒤 `--resume`하면 기존 3모델 run에 새 모델 결과가 섞이는데 manifest에는
  옛 3모델만 남아, "이 리포트는 어떤 구성으로 뽑혔나"가 거짓이 됐다.
"""

import pytest

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
