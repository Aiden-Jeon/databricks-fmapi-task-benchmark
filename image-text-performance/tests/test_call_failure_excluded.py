"""호출 실패(parsed=None)가 채점에서 제외되는지 — 전 태스크 계약 테스트.

2026-08-05 실측 사고: opus 엔드포인트가 `HTTP 502 "invalid response from an upstream
server"`를 내며 IMG-2에서 11/30 실패했는데, 그 실패가 **0점으로 채점**돼 micro_f1이
0.671로 나왔다(성공 19건만 보면 0.786). 같은 샘플을 sol은 전부 성공했으니 데이터가
아니라 엔드포인트 문제였다 — 즉 **장애가 모델 성능으로 오독**됐다.

원인은 태스크마다 실패 처리가 갈렸던 것:
- 분류 태스크(IMG-3/4/5, TXT-6/8): parse_output이 None → 이미 제외됨 ✅
- IMG-2: `"__ERROR__: ..."`를 정상 텍스트로 파싱해 **빈 태그셋 = 0점** ❌
- IMG-6·TXT-1/2/3/4: 누적기가 None을 유효 샘플로 세어 **0점** ❌
- TXT-5·TXT-7: None에 **예외**를 던져 셀 전체가 error ❌

조치: 러너가 `finish == "error"`면 parse_output을 건너뛰고 None을 쓴다(한 곳에서 일괄),
누적기들은 None을 분모에서 빼고 `n_skipped`로 노출한다.

이 테스트는 **14개 태스크 전부**에 대해 그 계약을 고정한다. 새 태스크를 추가하면
여기서 자동으로 검증된다(파라미터화).
"""

import pytest

from src.datasets_loader import load_registry
from src.tasks.base import Sample
from src.tasks.loader import discover_tasks

# 태스크별 reference 형태(정답 타입이 달라 누적기가 요구하는 형태를 맞춰준다)
_REFERENCE_BY_TASK = {
    "IMG-1": "a cat on a mat",
    "IMG-2": {"cat"},
    "IMG-6": "<table><tr><td>a</td></tr></table>",
    "TXT-1": ["answer"],
    "TXT-2": ["answer"],
    "TXT-3": "<table><tr><td>a</td></tr></table>",
    "TXT-4": ["정답"],
    "TXT-5": "요약문",
    "TXT-7": {"keyphrase"},
}


def _tasks_with_accumulator():
    registry = load_registry()
    out = []
    for tid, cls in sorted(discover_tasks().items()):
        inst = cls({}, registry)
        if hasattr(inst, "make_accumulator"):
            out.append((tid, inst))
    return out


def _sample(tid: str) -> Sample:
    return Sample(
        sample_id=0,
        inputs={},
        reference=_REFERENCE_BY_TASK.get(tid, 0),   # 기본 0 = 이진 분류 라벨
        lang="en",
    )


@pytest.mark.parametrize("tid,inst", _tasks_with_accumulator(), ids=lambda x: x if isinstance(x, str) else "")
def test_none_excluded_from_scoring(tid, inst):
    """호출 실패(None)만 들어오면 n_evaluated=0 — 0점 샘플로 세지 않는다."""
    acc = inst.make_accumulator()
    acc.add(None, _sample(tid))
    out = acc.finalize()
    assert out.get("n_evaluated") == 0, (
        f"{tid}: 호출 실패가 채점에 포함됐다(n_evaluated={out.get('n_evaluated')}). "
        f"엔드포인트 장애가 성능 저하로 오독된다."
    )


@pytest.mark.parametrize("tid,inst", _tasks_with_accumulator(), ids=lambda x: x if isinstance(x, str) else "")
def test_none_does_not_raise(tid, inst):
    """None에 예외를 던지지 않는다 — 던지면 셀 전체가 error로 죽어 정상 샘플까지 버려진다."""
    acc = inst.make_accumulator()
    acc.add(None, _sample(tid))
    acc.finalize()   # 예외 없이 통과해야 한다


def test_img2_parse_output_rejects_error_marker():
    """IMG-2가 `__ERROR__` 문자열을 태그로 파싱하지 않는다(이 사고의 직접 원인)."""
    registry = load_registry()
    inst = discover_tasks()["IMG-2"]({}, registry)
    s = _sample("IMG-2")
    assert inst.parse_output("__ERROR__: FMAPIError: HTTP 502", s) is None
    assert inst.parse_output("", s) is None
    # 정상 응답은 그대로 파싱
    assert inst.parse_output("cat, person", s) == {"cat", "person"}


def test_mixed_failure_and_success_uses_only_success():
    """실패와 성공이 섞이면 성공분만으로 평균 — 분모가 실패로 부풀지 않는다."""
    registry = load_registry()
    inst = discover_tasks()["IMG-2"]({}, registry)
    acc = inst.make_accumulator()

    s = Sample(sample_id=0, inputs={}, reference={"cat"}, lang="en")
    acc.add({"cat"}, s)      # 완전 정답
    acc.add(None, s)         # 호출 실패
    acc.add(None, s)         # 호출 실패
    out = acc.finalize()

    assert out["n_evaluated"] == 1
    assert out["micro_f1"] == pytest.approx(1.0), "실패가 0점으로 섞이면 0.33이 된다"
