"""실패 처리 2축 계약 테스트 — 호출 실패(제외) vs 파싱 실패(0점).

**두 실패는 다르게 취급해야 한다:**
- `CALL_FAILED`(HTTP 502·타임아웃) = 응답을 못 받음 → **인프라 문제** → 채점에서 제외.
  0점으로 세면 엔드포인트 장애가 성능 저하로 오독된다(2026-08-05 실측: opus IMG-2가
  502로 11/30 실패해 micro_f1 0.671, 성공 19건만 보면 0.786).
- `None`(형식 불일치) = 응답은 받았지만 파싱 실패 → **모델 능력 문제** → **0점 채점**.
  제외하면 형식을 대부분 못 맞추는 새 모델이 "성공한 일부"만으로 높은 점수를 받는다
  (2026-08-06 지적).

한동안 둘을 None으로 합쳐서 두 요구가 충돌했다. 지금은 러너가 호출 실패에만
`CALL_FAILED` sentinel을 넘기고, 누적기가 그것만 분모에서 뺀다(`n_skipped`로 노출).

이 테스트는 **14개 태스크 전부**에 대해 두 축을 고정한다(파라미터화 — 새 태스크 자동 검증).
"""

import pytest

from src.datasets_loader import load_registry
from src.scoring.accumulators import CALL_FAILED
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
def test_call_failure_excluded_from_scoring(tid, inst):
    """호출 실패(CALL_FAILED)만 들어오면 n_evaluated=0 — 0점 샘플로 세지 않는다."""
    acc = inst.make_accumulator()
    acc.add(CALL_FAILED, _sample(tid))
    out = acc.finalize()
    assert out.get("n_evaluated") == 0, (
        f"{tid}: 호출 실패가 채점에 포함됐다(n_evaluated={out.get('n_evaluated')}). "
        f"엔드포인트 장애가 성능 저하로 오독된다."
    )


@pytest.mark.parametrize("tid,inst", _tasks_with_accumulator(), ids=lambda x: x if isinstance(x, str) else "")
def test_neither_failure_kind_raises(tid, inst):
    """두 실패 모두 예외를 던지지 않는다 — 던지면 셀 전체가 error로 죽어 정상 샘플까지 버려진다."""
    for bad in (CALL_FAILED, None):
        acc = inst.make_accumulator()
        acc.add(bad, _sample(tid))
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
    acc.add(CALL_FAILED, s)  # 호출 실패
    acc.add(CALL_FAILED, s)  # 호출 실패
    out = acc.finalize()

    assert out["n_evaluated"] == 1
    assert out["micro_f1"] == pytest.approx(1.0), "실패가 0점으로 섞이면 0.33이 된다"


# ──────────────────────────────────────── 파싱 실패는 0점(능력 문제)


@pytest.mark.parametrize("tid,inst", _tasks_with_accumulator(), ids=lambda x: x if isinstance(x, str) else "")
def test_parse_failure_is_scored_as_zero(tid, inst):
    """파싱 실패(None)는 **분모에 포함**된다 — 형식을 못 맞춘 건 실제 능력 문제다.

    이걸 제외하면 형식을 대부분 못 맞추는 새 모델이 "성공한 일부"만으로 높은 점수를 받는다.
    (분류 태스크는 n_unparsed로 따로 세고 n_evaluated에서 빼는 기존 규약을 유지하므로,
     여기서는 '무언가로 집계됐는지'를 본다 — 조용히 사라지지 않아야 한다.)
    """
    acc = inst.make_accumulator()
    acc.add(None, _sample(tid))
    out = acc.finalize()

    counted = (out.get("n_evaluated") or 0) + (out.get("n_unparsed") or 0)
    assert counted == 1, (
        f"{tid}: 파싱 실패가 어디에도 집계되지 않았다(out={out}). "
        f"형식 불일치가 조용히 사라지면 모델이 부당하게 높은 점수를 받는다."
    )
    assert not out.get("n_skipped"), (
        f"{tid}: 파싱 실패를 호출 실패(n_skipped)로 집계했다 — 두 축이 섞였다"
    )


def test_mixed_failures_are_counted_separately():
    """호출 실패와 파싱 실패가 한 셀에 섞여도 각각 다른 칸으로 집계된다."""
    from src.tasks.loader import discover_tasks

    inst = discover_tasks()["TXT-1"]({}, load_registry())
    acc = inst.make_accumulator()
    s = _sample("TXT-1")
    acc.add("answer", s)        # 성공
    acc.add(None, s)            # 파싱 실패 → 0점으로 분모 포함
    acc.add(CALL_FAILED, s)     # 호출 실패 → 제외
    out = acc.finalize()

    assert out["n_evaluated"] == 2, f"파싱 실패가 분모에서 빠졌다: {out}"
    assert out["n_skipped"] == 1, f"호출 실패가 분모에 들어갔다: {out}"


# ──────────────────────────────── IMG-1 BERTScore 버퍼 (review 지적, 2026-08-06)


def test_img1_bertscore_excludes_call_failures():
    """IMG-1의 BERTScore 버퍼가 호출 실패를 담지 않는다.

    (Isaac Review 지적) `CALL_FAILED`는 falsy라 `str(parsed or "")`가 빈 문자열이 되어
    "빈 후보 vs 정답" 쌍이 BERTScore에 들어갔다. 그러면 caption_token_f1(호출 실패 제외)과
    bertscore(포함)의 분모가 어긋나고, 엔드포인트 장애가 BERTScore를 끌어내린다.
    파싱 실패(None)는 실제 0점이므로 **포함**돼야 한다 — 두 축을 여기서도 지킨다.
    """
    from src.tasks.loader import discover_tasks

    inst = discover_tasks()["IMG-1"]({}, load_registry())
    acc = inst.make_accumulator()
    s = Sample(sample_id=0, inputs={}, reference=["a cat on a mat"], lang="en")

    acc.add("a cat sits on a mat", s)   # 성공 → 버퍼 포함
    acc.add(CALL_FAILED, s)             # 호출 실패 → 버퍼 제외
    acc.add(None, s)                    # 파싱 실패 → 빈 후보로 포함(0점)

    pairs = acc._pairs
    assert len(pairs) == 2, f"호출 실패가 BERTScore 버퍼에 들어갔다: {pairs}"
    assert pairs[0][0] == "a cat sits on a mat"
    assert pairs[1][0] == "", "파싱 실패는 빈 후보로 채점돼야 한다"


def test_img1_caption_bertscore_helper_filters_call_failed():
    """score() 경로가 쓰는 헬퍼도 호출 실패를 걸러낸다(직접 호출 방어)."""
    from src.tasks.img_1 import _caption_bertscore

    s = Sample(sample_id=0, inputs={}, reference=["a cat"], lang="en")
    out = _caption_bertscore([CALL_FAILED, CALL_FAILED], [s, s])
    # 유효 쌍이 하나도 없으면 계산 불가로 표시된다(빈 후보로 0점을 만들지 않는다).
    assert "bertscore_f1" not in out, out
