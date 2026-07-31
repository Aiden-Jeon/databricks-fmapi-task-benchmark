"""
Language-aware tokenizer for Korean and English text.

This module provides language-aware tokenization following the methodology in
appendix "채점 방법론 상세" (P0) of plan.md. For Korean (ko), it uses KoNLPy's
Mecab morpheme tokenization, which is essential for accurate token-level metrics
like ROUGE and Token-F1. For English and other languages, whitespace tokenization
is used.

Korean morpheme tokenization is critical because Korean text has no spaces between
morphemes—whitespace splitting produces incorrect token sequences and silently
distorts metrics like ROUGE and Token-F1. Mecab solves this by splitting words
into grammatically meaningful morphemes.

System dependencies (required for Korean only):
- mecab: tokenizer engine
- mecab-ko-dic: Korean dictionary for Mecab

If these are unavailable, attempting to tokenize Korean will raise a clear
RuntimeError with installation instructions.
"""


def tokenize(text: str, lang: str) -> list[str]:
    """
    Tokenize text in a language-aware manner.

    For Korean (ko), uses KoNLPy Mecab morpheme tokenization to handle
    agglutinative morphology without whitespace markers. For English (en)
    and other languages, uses whitespace splitting.

    Args:
        text: The text to tokenize.
        lang: Language code ('ko' for Korean, 'en' for English, etc.).

    Returns:
        A list of tokens (morphemes for Korean, whitespace-split for others).

    Raises:
        RuntimeError: If lang=='ko' and Mecab/konlpy is not installed.
        ValueError: If text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text)}")

    if lang == "ko":
        try:
            from konlpy.tag import Mecab
        except ImportError as e:
            raise RuntimeError(
                "Korean tokenization requires KoNLPy and Mecab.\n"
                "Install system dependencies:\n"
                "  macOS: brew install mecab mecab-ko mecab-ko-dic\n"
                "  Ubuntu/Debian: sudo apt-get install mecab mecab-ko-dic\n"
                "  Then: pip install konlpy\n"
                "See README for details."
            ) from e

        try:
            mecab = Mecab()
        except Exception as e:
            raise RuntimeError(
                "Mecab initialization failed. Ensure mecab and mecab-ko-dic "
                "are correctly installed (see system dependency instructions)."
            ) from e

        # Mecab returns (morpheme, pos_tag) tuples; extract morphemes
        tokens = mecab.morphs(text)
        return tokens
    else:
        # For English and other languages, use whitespace splitting
        return text.split()
