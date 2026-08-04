"""Feature helpers for KoBEST SentiNeg (pure-python, no external deps)."""
import re
import unicodedata

CHO = list("\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e")
JUNG = list("\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159\u315a\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163")
JONG = [""] + list("\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b\u313c\u313d\u313e\u313f\u3140\u3141\u3142\u3144\u3145\u3146\u3147\u3148\u314a\u314b\u314c\u314d\u314e")

_SBASE = 0xAC00
_SLAST = 0xD7A3


def decompose(text: str) -> str:
    """Decompose Hangul syllables into jamo sequence (keeps other chars as-is).

    Each syllable becomes cho+jung+jong (+ '-' when there is no final consonant)
    so that positions stay aligned for character n-grams.
    """
    out = []
    for ch in text:
        code = ord(ch)
        if _SBASE <= code <= _SLAST:
            idx = code - _SBASE
            out.append(CHO[idx // 588])
            out.append(JUNG[(idx % 588) // 28])
            j = JONG[idx % 28]
            out.append(j if j else "-")
        else:
            out.append(ch)
    return "".join(out)


_WS = re.compile(r"\s+")
_REP = re.compile(r"(.)\1{2,}")


def normalize(text: str) -> str:
    """Light normalisation: NFC, collapse whitespace / long char repeats."""
    text = unicodedata.normalize("NFC", str(text))
    text = _REP.sub(r"\1\1", text)
    text = _WS.sub(" ", text).strip()
    return text


def jamo_norm(text: str) -> str:
    return decompose(normalize(text))
