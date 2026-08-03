"""Shared feature-engineering utilities for KorFin-ASC."""
import re
import numpy as np
import pandas as pd

MARK_L = "《"
MARK_R = "》"
TGT = "㋣"  # single-char placeholder for the target aspect


def _find_all(hay: str, needle: str):
    out = []
    if not needle:
        return out
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def mark_sentence(sent: str, aspect: str) -> str:
    """Wrap every occurrence of the aspect with explicit marker characters."""
    if not isinstance(sent, str):
        sent = ""
    if not isinstance(aspect, str) or not aspect:
        return sent
    if aspect in sent:
        return sent.replace(aspect, MARK_L + aspect + MARK_R)
    return MARK_L + aspect + MARK_R + " : " + sent


def mask_sentence(sent: str, aspect: str) -> str:
    """Replace the aspect with a generic placeholder (aspect-identity agnostic)."""
    if not isinstance(sent, str):
        sent = ""
    if not isinstance(aspect, str) or not aspect:
        return sent
    if aspect in sent:
        return sent.replace(aspect, TGT)
    return TGT + " : " + sent


def context_window(sent: str, aspect: str, width: int = 30) -> str:
    """Text around the first/last occurrence of the aspect, aspect itself masked."""
    if not isinstance(sent, str):
        sent = ""
    if not isinstance(aspect, str) or not aspect or aspect not in sent:
        return TGT + " " + sent[:2 * width]
    idxs = _find_all(sent, aspect)
    chunks = []
    for i in idxs[:3]:
        lo = max(0, i - width)
        hi = min(len(sent), i + len(aspect) + width)
        chunks.append(sent[lo:i] + TGT + sent[i + len(aspect):hi])
    return " ".join(chunks)


def clause_window(sent: str, aspect: str) -> str:
    """The clause (split on Korean sentence/clause delimiters) containing the aspect."""
    if not isinstance(sent, str):
        sent = ""
    if not isinstance(aspect, str) or not aspect or aspect not in sent:
        return TGT + " " + sent
    parts = re.split(r"(?<=[.!?,;])\s+|(?<=[다요음])\s+(?=[가-힣])", sent)
    hit = [p for p in parts if aspect in p]
    if not hit:
        hit = [sent]
    return " ".join(p.replace(aspect, TGT) for p in hit)


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    s = df["sentence"].fillna("").astype(str)
    a = df["aspect"].fillna("").astype(str)
    out["marked"] = [mark_sentence(x, y) for x, y in zip(s, a)]
    out["masked"] = [mask_sentence(x, y) for x, y in zip(s, a)]
    out["ctx"] = [context_window(x, y, 30) for x, y in zip(s, a)]
    out["ctx_s"] = [context_window(x, y, 12) for x, y in zip(s, a)]
    out["clause"] = [clause_window(x, y) for x, y in zip(s, a)]
    out["aspect"] = a
    out["sentence"] = s
    return out


def numeric_feats(df: pd.DataFrame) -> np.ndarray:
    s = df["sentence"].fillna("").astype(str)
    a = df["aspect"].fillna("").astype(str)
    n_occ = np.array([len(_find_all(x, y)) if y else 0 for x, y in zip(s, a)], dtype=float)
    pos = np.array([(x.find(y) / max(len(x), 1)) if (y and y in x) else -1.0
                    for x, y in zip(s, a)], dtype=float)
    feats = np.column_stack([
        s.str.len().values / 100.0,
        a.str.len().values / 10.0,
        n_occ,
        pos,
        s.str.count(r"[%]").values,
        s.str.count(r"\d").values / 10.0,
        s.str.contains(r"상승|급등|호조|증가|개선|성장|수혜|기대|양호|확대|사상 최대|흑자").astype(float).values,
        s.str.contains(r"하락|급락|부진|감소|악화|우려|손실|적자|둔화|축소|리스크|무산").astype(float).values,
        s.str.contains(r"목표주가|투자의견|매수|비중확대|Buy|BUY").astype(float).values,
        (a.str.len() == 0).astype(float).values,
    ])
    return feats
