"""Data loading / entity-string parsing / char-BIO alignment for KLUE-NER task."""
import pandas as pd

LABELS = ["PS", "LC", "OG", "DT", "TI", "QT"]
LABSET = set(LABELS)


def parse_entities(s):
    """Parse 'surface:TYPE|surface:TYPE' -> [(surface, type), ...] robustly.

    Surfaces may themselves contain ':' (e.g. '9:30') or '|' (rare), so we
    split on '|' and re-merge segments whose tail is not a valid label.
    """
    if not isinstance(s, str) or s.strip() == "":
        return []
    segs = s.split("|")
    merged = []
    for seg in segs:
        if ":" in seg and seg.rsplit(":", 1)[1] in LABSET:
            merged.append(seg)
        elif merged:
            merged[-1] = merged[-1] + "|" + seg
        else:
            merged.append(seg)
    out = []
    for seg in merged:
        if ":" not in seg:
            continue
        surf, typ = seg.rsplit(":", 1)
        if typ in LABSET and surf != "":
            out.append((surf, typ))
    return out


def format_entities(ents):
    return "|".join("%s:%s" % (s, t) for s, t in ents)


def align(sentence, ents):
    """Greedy left-to-right alignment of entity surfaces onto the sentence.

    Returns (spans, n_failed) where spans = [(start, end, type)] sorted, non-overlapping.
    """
    spans = []
    used = [False] * len(sentence)
    cursor = 0
    failed = 0
    for surf, typ in ents:
        pos = -1
        # try from cursor first, then from 0, skipping already-used positions
        for start_from in (cursor, 0):
            p = sentence.find(surf, start_from)
            while p != -1:
                if not any(used[p:p + len(surf)]):
                    pos = p
                    break
                p = sentence.find(surf, p + 1)
            if pos != -1:
                break
        if pos == -1:
            failed += 1
            continue
        for i in range(pos, pos + len(surf)):
            used[i] = True
        spans.append((pos, pos + len(surf), typ))
        cursor = pos + len(surf)
    spans.sort()
    return spans, failed


def spans_to_bio(n, spans):
    tags = ["O"] * n
    for s, e, t in spans:
        tags[s] = "B-" + t
        for i in range(s + 1, e):
            tags[i] = "I-" + t
    return tags


def bio_to_ents(sentence, tags):
    """Decode BIO tag sequence into ordered [(surface, type)] list."""
    ents = []
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t.startswith("B-"):
            typ = t[2:]
            j = i + 1
            while j < n and tags[j] == "I-" + typ:
                j += 1
            surf = sentence[i:j].strip()
            if surf:
                ents.append((surf, typ))
            i = j
        elif t.startswith("I-"):  # stray I- treated as B-
            typ = t[2:]
            j = i + 1
            while j < n and tags[j] == "I-" + typ:
                j += 1
            surf = sentence[i:j].strip()
            if surf:
                ents.append((surf, typ))
            i = j
        else:
            i += 1
    return ents


def load(path, with_labels=True):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    rows = []
    total_ent = total_fail = 0
    for r in df.itertuples(index=False):
        sent = r.sentence
        item = {"id": r.id, "sentence": sent}
        if with_labels:
            ents = parse_entities(r.entities)
            spans, failed = align(sent, ents)
            total_ent += len(ents)
            total_fail += failed
            item["ents"] = ents
            item["spans"] = spans
            item["tags"] = spans_to_bio(len(sent), spans)
        rows.append(item)
    if with_labels:
        return rows, total_ent, total_fail
    return rows


def micro_f1(gold_list, pred_list):
    """Entity-level micro F1 with multiset (count) matching per sentence."""
    from collections import Counter
    tp = fp = fn = 0
    for g, p in zip(gold_list, pred_list):
        gc, pc = Counter(g), Counter(p)
        inter = sum((gc & pc).values())
        tp += inter
        fp += sum(pc.values()) - inter
        fn += sum(gc.values()) - inter
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    rows, te, tf = load("train.csv")
    print("sentences", len(rows), "entities", te, "align_failed", tf)
    # round-trip check
    ok = 0
    golds, preds = [], []
    for r in rows:
        rec = bio_to_ents(r["sentence"], r["tags"])
        golds.append(r["ents"])
        preds.append(rec)
        if rec == r["ents"]:
            ok += 1
    print("roundtrip exact sentences", ok, "/", len(rows))
    print("roundtrip f1 (ceiling)", micro_f1(golds, preds))
