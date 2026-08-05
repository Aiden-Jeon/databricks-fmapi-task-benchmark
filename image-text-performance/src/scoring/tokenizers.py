"""언어별 토크나이저 (plan 부록 P0).

한국어는 교착어라 공백 분리로 토큰화하면 ROUGE·Token-F1이 조용히 왜곡된다.
따라서 형태소 분석기(KoNLPy Mecab)를 쓴다. 영어 등은 공백 분리.

Mecab은 시스템 의존성(mecab, mecab-ko-dic)이 필요하다. 미설치 환경에서 벤치마크가
죽지 않도록, Mecab이 없으면 **음절 단위(syllable)** fallback으로 자동 전환한다.
음절 단위는 공백 분리보다 훨씬 정확하지만 형태소만큼은 아니므로, 어느 방식이 쓰였는지
`korean_tokenizer_backend()`로 조회해 리포트에 명시할 수 있다.

형태소 백엔드 설치 (2026-08-05 실측 정리):
- **권장(가장 쉬움)**: `pip install mecab-ko mecab-ko-dic` — 사전까지 wheel에 들어 있어
  시스템 패키지가 필요 없다. Apple Silicon에서 이것만으로 동작 확인.
- KoNLPy 경로: `brew install mecab-ko mecab-ko-dic` + konlpy. 단 함정이 많다 —
  (1) `brew install mecab`(일본어 ipadic)과 `mecab-ko`가 충돌해 `brew unlink mecab` 필요,
  (2) `/opt/homebrew/etc/mecabrc`의 dicdir이 없는 ipadic을 가리켜 수동 수정 필요,
  (3) konlpy의 기본 dicpath가 Intel Mac 기준(`/usr/local/...`)이라 Apple Silicon에서 빗나감,
  (4) `pip install mecab-python3`는 자체 libmecab을 번들해 시스템 mecab-ko 사전과 ABI 불일치.
- Ubuntu/Debian: `sudo apt-get install mecab libmecab-dev mecab-ko-dic` 또는 위 pip 경로.
"""

from __future__ import annotations

import re

# Mecab 인스턴스 캐시 (최초 1회만 초기화 시도)
_mecab = None
_mecab_tried = False
_backend = "unknown"  # "mecab" | "syllable"


class _MecabKo:
    """`mecab-ko` + `mecab-ko-dic`(pip) 래퍼. konlpy와 같은 `.morphs()` 인터페이스 제공.

    konlpy 경로가 환경에 따라 잘 깨지므로(모듈 docstring 참고) 이 순수 pip 경로를
    **먼저** 시도한다. 파싱 결과 각 줄은 "표층형\\t품사,..." 이고 마지막에 EOS가 온다.
    """

    def __init__(self) -> None:
        import mecab_ko
        import mecab_ko_dic

        self._tagger = mecab_ko.Tagger(mecab_ko_dic.MECAB_ARGS)

    def morphs(self, text: str) -> list[str]:
        out = []
        for line in self._tagger.parse(text).split("\n"):
            if not line or line == "EOS":
                continue
            surface = line.split("\t", 1)[0]
            if surface:
                out.append(surface)
        return out


def _get_mecab():
    """Mecab을 지연 초기화. 실패하면 None (fallback 신호).

    두 경로를 순서대로 시도한다:
    1) `mecab_ko`(pip, 사전 포함) — 시스템 의존성이 없어 가장 잘 붙는다.
    2) `konlpy.tag.Mecab` — 시스템 mecab-ko-dic이 설치된 환경.
    둘 다 실패하면 음절 fallback(채점이 죽지는 않지만 형태소 기준이 아님).
    """
    global _mecab, _mecab_tried, _backend
    if _mecab_tried:
        return _mecab
    _mecab_tried = True

    try:
        m = _MecabKo()
        m.morphs("테스트")  # 실제 동작 확인 (사전 없으면 여기서 예외)
        _mecab, _backend = m, "mecab"
        return _mecab
    except Exception:
        pass

    try:
        from konlpy.tag import Mecab

        m = Mecab()
        m.morphs("테스트")  # 실제 동작 확인 (백엔드 없으면 여기서 예외)
        _mecab = m
        _backend = "mecab"
    except Exception:
        _mecab = None
        _backend = "syllable"
    return _mecab


# 한글 음절 + 영숫자 토큰 (음절 fallback용)
_SYLLABLE_RE = re.compile(r"[가-힣]|[a-zA-Z0-9]+")


def _syllable_tokenize(text: str) -> list[str]:
    """한글은 음절 하나씩, 영숫자 덩어리는 하나로. 형태소 대체 fallback."""
    return _SYLLABLE_RE.findall(text)


def korean_tokenizer_backend() -> str:
    """현재 한국어 토큰화에 쓰이는 백엔드: 'mecab' | 'syllable' | 'unknown'.

    리포트에 어떤 방식으로 채점했는지 명시하기 위함. tokenize(ko) 최초 호출 후 확정.
    """
    return _backend


def tokenize(text: str, lang: str) -> list[str]:
    """언어별 토큰화.

    - ko: Mecab 형태소. 없으면 음절 단위 fallback(자동).
    - en/기타: 공백 분리.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text)}")

    if lang == "ko":
        mecab = _get_mecab()
        if mecab is not None:
            return mecab.morphs(text)
        return _syllable_tokenize(text)
    return text.split()
