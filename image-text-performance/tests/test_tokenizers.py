"""한국어 토큰화 테스트 (이슈 E).

한국어는 교착어라 공백/음절 토큰화로 ROUGE·Token-F1을 재면 조용히 왜곡된다.
2026-08-05까지 mecab 백엔드가 안 붙어 **음절 폴백**으로 채점하고 있었다
(`'오늘은'` → `['오','늘','은']`). 형태소 백엔드가 붙으면 `['오늘','은']`이다.

이 테스트는 백엔드 유무와 무관하게 통과해야 한다(CI/타 환경엔 mecab이 없을 수 있음).
mecab이 있으면 형태소 분해를 확인하고, 없으면 폴백이 안전하게 동작함을 확인한다.
"""

import pytest

from src.scoring.tokenizers import korean_tokenizer_backend, tokenize


def _backend():
    tokenize("초기화", "ko")  # 백엔드 확정(지연 초기화)
    return korean_tokenizer_backend()


def test_english_is_whitespace_split():
    assert tokenize("the quick brown fox", "en") == ["the", "quick", "brown", "fox"]


def test_korean_tokenization_is_not_empty():
    """백엔드가 무엇이든 토큰이 나와야 한다(채점이 0으로 죽지 않도록)."""
    assert tokenize("오늘은 날씨가 좋다", "ko")


def test_backend_is_known_value():
    assert _backend() in {"mecab", "syllable"}


@pytest.mark.skipif(_backend() != "mecab", reason="mecab 백엔드 미설치 환경")
def test_mecab_splits_morphemes_not_syllables():
    """형태소 분해 확인 — 조사가 어간에서 분리되고 어간이 음절로 쪼개지지 않는다."""
    toks = tokenize("오늘은 날씨가 좋다", "ko")
    assert "오늘" in toks and "날씨" in toks
    assert "은" in toks and "가" in toks      # 조사 분리
    assert "늘" not in toks                    # 음절 폴백이면 나타나는 토큰


@pytest.mark.skipif(_backend() != "mecab", reason="mecab 백엔드 미설치 환경")
def test_mecab_scoring_credits_partial_answer():
    """형태소 기준이면 '베토벤의 합창교향곡' vs '합창교향곡'이 부분 점수를 받는다.

    음절 폴백에서는 글자 겹침으로 점수가 부풀고, 공백 분리에서는 조사가 붙어
    (`합창교향곡` vs `합창교향곡`이 아닌 `베토벤의`) 매칭이 어긋난다.
    """
    from src.scoring.metrics import token_f1

    score = token_f1("베토벤의 합창교향곡", "합창교향곡", "ko")
    assert 0.0 < score < 1.0


def test_syllable_fallback_shape_when_no_mecab():
    """폴백 환경에서는 음절 단위여야 한다(형태소가 아니어도 채점은 가능해야 함)."""
    if _backend() == "mecab":
        pytest.skip("mecab 백엔드가 붙은 환경")
    assert tokenize("오늘은", "ko") == ["오", "늘", "은"]


def test_empty_input_safe():
    assert tokenize("", "ko") == []
    assert tokenize("", "en") == []
