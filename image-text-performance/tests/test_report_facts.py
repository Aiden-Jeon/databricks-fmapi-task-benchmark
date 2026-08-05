"""리포트 집계 테스트 — 실패 셀 제외(이슈 H)와 동점 공동 1위(이슈 I).

두 항목 모두 "수치가 실제와 다른 결론을 만들던" 버그의 회귀 방지용이다:
- H: 호출이 대량 실패한 셀의 0점이 "성능 낮음"으로 집계돼 순위가 뒤집혔다
  (실측 IMG-6: opus 19/30 실패 시 cell_f1 0.290 → sol 승. 재실행하면 0.841로 opus 승).
- I: 동점일 때 dict 첫 모델에만 승리를 줘서 1위 횟수가 부풀었다.
"""

from src.report.generate import _extract_facts, _quant_table


def _cell(task, model, value, *, metric="cell_f1", **extra):
    """scores.json 한 셀 모양을 만든다(러너가 쓰는 구조와 동일)."""
    entry = {
        "model_id": model,
        "task_id": task,
        "reasoning_mode": "minimal",
        "n": 30,
        "metrics": {metric: value},
    }
    entry.update(extra)
    return entry


def _scores(*cells):
    return {f"{c['model_id']}::{c['task_id']}::minimal": c for c in cells}


# ---------------------------------------------------------------- 이슈 H


def test_unreliable_cell_excluded_from_ranking():
    """실패율 과다(unreliable) 셀은 순위 집계에서 빠진다.

    IMG-6 실측 상황 재현: opus가 19/30 실패해 0.29로 나오면, 그 값을 그대로 쓰면
    sol(0.76)이 이긴 것처럼 집계된다. unreliable이면 opus 점수를 아예 쓰지 않는다.
    """
    scores = _scores(
        _cell("IMG-6", "opus", 0.29, n_call_failed=19, call_failure_rate=0.633, unreliable=True),
        _cell("IMG-6", "sol", 0.76),
    )
    facts = _extract_facts(scores, perf={})

    assert "opus" not in facts["per_task_scores"]["IMG-6/minimal"]
    assert facts["win_counts"] == {"sol": 1}
    assert facts["excluded_unreliable"] == ["IMG-6/opus"]


def test_partial_failure_below_threshold_still_counted():
    """실패가 임계 미만이면 점수는 그대로 집계한다(과잉 제외 방지)."""
    scores = _scores(
        _cell("IMG-6", "opus", 0.84, n_call_failed=2, call_failure_rate=0.0667),
        _cell("IMG-6", "sol", 0.79),
    )
    facts = _extract_facts(scores, perf={})

    assert facts["win_counts"] == {"opus": 1}
    assert facts["excluded_unreliable"] == []


def test_quant_table_shows_failure_column():
    """정량표에 호출·judge 실패가 보인다 — 점수만 보고 오독하지 않도록."""
    scores = _scores(
        _cell("IMG-6", "opus", 0.29, n_call_failed=19, call_failure_rate=0.633, unreliable=True),
        _cell("TXT-5", "sol", 0.31, metric="rouge1"),
    )
    scores["sol::TXT-5::minimal"]["metrics"]["judge_detail"] = {"n": 29, "n_failed": 1}

    table = _quant_table(scores, task_labels={})

    assert "실패" in table.splitlines()[0]      # 헤더에 실패 열
    assert "호출 19/30" in table
    assert "신뢰불가" in table                   # unreliable 경고
    assert "judge 1" in table


def test_quant_table_marks_clean_cells_with_dash():
    """실패가 없으면 실패 열은 '—' (빈칸으로 두면 표가 어긋난다)."""
    table = _quant_table(_scores(_cell("TXT-3", "sol", 0.98)), task_labels={})
    assert table.strip().endswith("| — |")


# ---------------------------------------------------------------- 이슈 I


def test_tie_counted_as_joint_winners():
    """동점은 공동 1위 — 옛 구현은 dict 첫 모델에만 승리를 줬다."""
    scores = _scores(
        _cell("TXT-6", "opus", 0.833),
        _cell("TXT-6", "sol", 0.833),
        _cell("TXT-6", "glm", 0.5),
    )
    facts = _extract_facts(scores, perf={})

    assert facts["task_winners"]["TXT-6/minimal"] == ["opus", "sol"]
    assert facts["win_counts"] == {"opus": 1, "sol": 1}
    assert facts["n_tied_tasks"] == 1


def test_no_tie_gives_single_winner():
    """동점이 아니면 단독 1위 한 명만."""
    scores = _scores(_cell("TXT-4", "glm", 0.951), _cell("TXT-4", "opus", 0.837))
    facts = _extract_facts(scores, perf={})

    assert facts["task_winners"]["TXT-4/minimal"] == ["glm"]
    assert facts["n_tied_tasks"] == 0


def test_tie_order_is_deterministic():
    """공동 1위 목록은 정렬돼 있어 리포트가 실행마다 흔들리지 않는다."""
    scores = _scores(
        _cell("TXT-8", "sol", 0.7),
        _cell("TXT-8", "glm", 0.7),
        _cell("TXT-8", "opus", 0.7),
    )
    facts = _extract_facts(scores, perf={})
    assert facts["task_winners"]["TXT-8/minimal"] == ["glm", "opus", "sol"]


def test_error_cells_ignored():
    """채점 예외(metrics.error) 셀은 기존대로 무시한다."""
    scores = _scores(_cell("TXT-1", "opus", 0.9))
    scores["sol::TXT-1::minimal"] = {
        "model_id": "sol", "task_id": "TXT-1", "reasoning_mode": "minimal",
        "n": 30, "metrics": {"error": "ValueError: boom"},
    }
    facts = _extract_facts(scores, perf={})
    assert facts["win_counts"] == {"opus": 1}
