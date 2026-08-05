"""LLM-judge 점수 파싱 테스트.

judge 파싱은 **조용히 틀리는** 종류의 코드다: 점수를 못 찾았는데 아무 숫자를 주워오면
리포트의 judge 평균이 그럴듯한 값으로 오염되고, 원문이 저장되지 않아 사후 검증도 안 된다
(2026-08-05 실측된 실제 버그). 그래서 "인정해야 할 형태"와 "거부해야 할 형태"를 함께 고정한다.
"""

import pytest

from src.scoring.judge import (
    JUDGE_MAX_TOKENS,
    build_judge_prompt,
    parse_judge_score,
    run_judge,
    summarize_judge_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# 인정: 명시적 점수 표현
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Score: 4", 4),
        ("score: 4.5", 4),                      # 소수는 절삭
        ("Rating: 3", 3),
        ("score is 5", 5),
        ("**Final score:** 5", 5),              # 마크다운 장식
        ("**Score:** 2", 2),
        ("점수: 3", 3),                          # 한국어 라벨
        ("Reasoning blah.\nScore: 1", 1),       # 근거 뒤 점수
        ("4/5", 4),
        ("4 out of 5", 4),
        ("4", 4),                                # 응답이 숫자뿐
        ("**5**", 5),
        ("4.", 4),
    ],
)
def test_parse_accepts_explicit_scores(text, expected):
    assert parse_judge_score(text) == expected


def test_parse_uses_last_score_when_rubric_quoted_first():
    """루브릭을 인용한 뒤 최종 판정을 말하는 경우 — 뒤쪽(최종)을 쓴다."""
    assert parse_judge_score("anchor 3 says X. Final Score: 5") == 5


@pytest.mark.parametrize("text,expected", [("Score: 0", 1), ("Score: 7", 5), ("Rating: 12", 5)])
def test_parse_clamps_out_of_range_scores(text, expected):
    """judge가 범위를 벗어난 점수를 명시하면 버리지 않고 [1,5]로 절단."""
    assert parse_judge_score(text) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 거부: 점수가 아닌 숫자 (옛 오탐 — 회귀 금지)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text",
    [
        # judge 응답이 max_tokens로 잘린 실제 사례
        "**Reasoning:**\nThe candidate output captures all the core",
        # 이하 3건은 옛 파서가 각각 3 / 4 / 5로 오인했던 문장
        "**Reasoning:**\nThe caption accurately describes 2 sinks and 3 mirrors",
        "The candidate matches anchor 4 description but",
        "**Reasoning:** It captures 1 of the 5 key elements",
        # usage 로그 같은 무관한 숫자
        "prompt_tokens 285 completion 12",
        # 숫자가 아예 없는 서술
        "The answer is correct and complete.",
    ],
)
def test_parse_rejects_prose_numbers(text):
    """본문 숫자를 점수로 지어내지 않는다 — 실패는 None으로 드러낸다."""
    assert parse_judge_score(text) is None


@pytest.mark.parametrize("text", ["", None])
def test_parse_handles_empty_input(text):
    assert parse_judge_score(text) is None


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트·상수
# ─────────────────────────────────────────────────────────────────────────────
def test_judge_max_tokens_is_large_enough_for_reasoning():
    """gemini는 reasoning을 못 끄므로 사고 240토큰 + 근거 서술을 담을 여유가 필요.

    실측: 256이면 사고에 소진돼 본문이 잘렸다(finish_reason=length → 파싱 실패).
    """
    assert JUDGE_MAX_TOKENS >= 512


def test_build_judge_prompt_embeds_yaml_rubric_scale():
    """config/judge_rubrics.yaml 스키마({task, language, scale})의 앵커가 프롬프트에 들어가야 한다.

    옛 버그: name/anchors만 읽어 YAML의 scale이 조용히 무시되고 채점 기준 없는
    프롬프트가 전송됐다.
    """
    rubric = {"task": "이미지 캡션 정확도", "language": "en", "scale": {1: "나쁨", 5: "완벽"}}
    prompt = build_judge_prompt("IMG-1", "Q?", "ref", "cand", rubric)

    assert "이미지 캡션 정확도" in prompt
    assert "나쁨" in prompt and "완벽" in prompt
    assert "ref" in prompt and "cand" in prompt


def test_build_judge_prompt_accepts_inline_fallback_schema():
    """태스크 인라인 fallback 스키마({name, description, anchors})도 지원."""
    rubric = {"name": "Doc QA", "description": "설명", "anchors": {"1": "poor", "5": "great"}}
    prompt = build_judge_prompt("TXT-1", "Q?", "ref", "cand", rubric)

    assert "Doc QA" in prompt
    assert "poor" in prompt and "great" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# run_judge / summarize_judge_scores — 실패를 값으로 메우지 않는 계약
# ─────────────────────────────────────────────────────────────────────────────
class _FakeResponse:
    def __init__(self, text, finish_reason="stop"):
        self.text = text
        self.finish_reason = finish_reason


class _FakeClient:
    """judge_client 대역. chat()이 미리 정한 응답을 주거나 예외를 던진다."""

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def chat(self, *, endpoint, messages, max_tokens):
        self.calls.append({"endpoint": endpoint, "max_tokens": max_tokens})
        if self._raises:
            raise self._raises
        return self._response


def test_run_judge_returns_parsed_score():
    client = _FakeClient(_FakeResponse("Reasoning...\nScore: 4"))
    assert run_judge(client, "judge-ep", "prompt", "TXT-1", 0) == 4


def test_run_judge_uses_shared_max_tokens():
    """태스크가 각자 max_tokens를 정하지 않고 공용 상수를 쓴다(잘림 재발 방지)."""
    client = _FakeClient(_FakeResponse("Score: 3"))
    run_judge(client, "judge-ep", "prompt", "TXT-1", 0)
    assert client.calls[0]["max_tokens"] == JUDGE_MAX_TOKENS


def test_run_judge_returns_none_on_truncated_response():
    """잘린 응답(finish=length)은 점수를 지어내지 않고 None."""
    client = _FakeClient(_FakeResponse("**Reasoning:**\nThe candidate captures all the core", "length"))
    assert run_judge(client, "judge-ep", "prompt", "IMG-1", 7) is None


def test_run_judge_returns_none_on_call_failure():
    """호출 예외(403·타임아웃 등)도 None — 3점으로 메우지 않는다."""
    client = _FakeClient(raises=RuntimeError("HTTP 403 blocked by IP ACL"))
    assert run_judge(client, "judge-ep", "prompt", "TXT-2", 3) is None


def test_summarize_excludes_failures_from_mean():
    """None은 평균 분모에서 빠진다. (4+2)/2 = 3.0, n_judged=2, 실패 2."""
    out = summarize_judge_scores([4, None, 2, None])
    assert out["judge_mean"] == pytest.approx(3.0)
    assert out["n_judged"] == 2
    assert out["n_judge_failed"] == 2


def test_summarize_reports_none_mean_when_all_failed():
    """전부 실패면 judge_mean=None — 0.0은 '최악 판정'으로 오독된다(옛 IMG-1 버그)."""
    out = summarize_judge_scores([None, None, None])
    assert out["judge_mean"] is None
    assert out["n_judged"] == 0
    assert out["n_judge_failed"] == 3


def test_summarize_keeps_raw_scores_for_audit():
    """원 점수 리스트(None 포함)를 보존해 사후에 실패 위치를 추적할 수 있다."""
    scores = [5, None, 1]
    assert summarize_judge_scores(scores)["judge_scores"] == scores


def test_summarize_handles_empty_input():
    out = summarize_judge_scores([])
    assert out["judge_mean"] is None and out["n_judged"] == 0
