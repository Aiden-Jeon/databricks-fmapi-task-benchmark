"""정량 메트릭 테스트 — ANLS(DocVQA 공식) 중심.

ANLS는 TXT-1의 대표 메트릭이라 회귀가 곧 리포트 왜곡이다. 공식 정의
(Biten et al., ICCV 2019: τ=0.5, 다중정답 max, 정규화 편집거리)의 성질을
케이스로 고정한다. 기대값은 손으로 계산 가능한 것만 사용한다.
"""

import pytest

from src.scoring.metrics import ANLS_THRESHOLD, anls


def test_anls_exact_match_is_one():
    """완전 일치는 1.0."""
    assert anls("485", ["485"]) == 1.0


def test_anls_case_and_space_insensitive():
    """정규화(strip+lowercase) 후 비교 — 대소문자·주변 공백은 감점 없음."""
    assert anls("  MARCH 27, 1979  ", ["March 27, 1979"]) == 1.0


def test_anls_takes_best_of_multiple_golds():
    """다중정답이면 최고 점수(공식 정의). '$485'가 있으므로 1.0."""
    assert anls("$485", ["485", "$485"]) == 1.0


def test_anls_tolerates_single_char_difference():
    """한 글자 차이는 관용(exact_match와 다른 지점). '485' vs '$485' = 1 - 1/4."""
    assert anls("485", ["$485"]) == pytest.approx(0.75)


def test_anls_thresholds_unrelated_answer_to_zero():
    """유사도가 τ 미만이면 0으로 절단 — 우연한 부분일치에 점수를 주지 않는다."""
    assert anls("completely different answer", ["485"]) == 0.0


def test_anls_threshold_boundary_is_inclusive():
    """유사도가 정확히 τ면 통과(>=). 'abcd' vs 'abxy' = 1 - 2/4 = 0.5."""
    assert anls("abcd", ["abxy"]) == pytest.approx(ANLS_THRESHOLD)


def test_anls_empty_prediction_scores_zero():
    """빈 예측은 0점(모델이 답을 못 낸 경우)."""
    assert anls("", ["485"]) == 0.0


def test_anls_both_empty_scores_one():
    """정답도 예측도 비면 1.0(비교 대상 없음)."""
    assert anls("", [""]) == 1.0


def test_anls_accepts_bare_string_gold():
    """gold가 리스트가 아닌 단일 문자열이어도 동작."""
    assert anls("485", "485") == 1.0


def test_anls_custom_threshold_disables_truncation():
    """threshold=0이면 절단 없이 원 유사도를 반환."""
    assert anls("abcd", ["abxy"], threshold=0.0) == pytest.approx(0.5)


def test_anls_is_symmetric_in_length_normalization():
    """정규화 분모가 max(len)이라 긴 쪽/짧은 쪽 순서에 무관."""
    assert anls("abc", ["abcdef"]) == anls("abcdef", ["abc"])


def test_anls_ignores_gold_that_is_empty_when_pred_present():
    """정답 리스트에 빈 문자열이 섞여 있어도 유효 정답으로 채점된다."""
    assert anls("485", ["", "485"]) == 1.0
