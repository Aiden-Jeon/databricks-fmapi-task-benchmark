"""Feature building for KLUE-RE with entity markers.

The entity spans (subject_entity / object_entity strings) always occur inside
the sentence, so we wrap their *first* occurrences with special marker tokens.
Marker position relative to the entities carries most of the signal for RE.
"""
import re

MARK = {"ss": "엔에스에스", "se": "엔에스이", "os": "엔오에스", "oe": "엔오이"}


def _escape(s: str) -> str:
    return re.escape(str(s))


def mark_entities(sentence: str, subj: str, obj: str) -> str:
    """Wrap first occurrences of subj / obj with marker tokens."""
    sent = str(sentence)
    s, o = str(subj), str(obj)
    # find first occurrence of each entity
    spans = []  # (start, end, kind) kind: 0=subj, 1=obj
    if s:
        i = sent.find(s)
        if i >= 0:
            spans.append((i, i + len(s), 0))
    if o:
        i = sent.find(o)
        if i >= 0:
            spans.append((i, i + len(o), 1))
    if not spans:
        return sent
    # drop overlaps (keep earlier/longer first-found)
    spans.sort()
    kept = []
    last_end = -1
    for st, en, k in spans:
        if st >= last_end:
            kept.append((st, en, k))
            last_end = en
    # rebuild from the end so offsets stay valid
    for st, en, k in reversed(kept):
        if k == 0:
            wrap = f" {MARK['ss']} {sent[st:en]} {MARK['se']} "
        else:
            wrap = f" {MARK['os']} {sent[st:en]} {MARK['oe']} "
        sent = sent[:st] + wrap + sent[en:]
    return sent


def build_text(df, sep=" [SEP] "):
    sents = [
        mark_entities(sent, s, o)
        for sent, s, o in zip(df["sentence"], df["subject_entity"], df["object_entity"])
    ]
    subj = df["subject_entity"].astype(str)
    obj = df["object_entity"].astype(str)
    return [f"{m}{sep}{s}{sep}{o}" for m, s, o in zip(sents, subj, obj)]
