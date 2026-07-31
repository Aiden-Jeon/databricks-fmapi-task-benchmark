"""Experiment: gazetteer post-processing to boost recall.

For each prediction, additionally insert entity spans that exactly match a
high-confidence gazetteer expression (single type, count >= threshold) and are
NOT already covered by a predicted entity of the same type. We evaluate on the
held-out split to choose a threshold; if it improves F1 we apply to test.
"""
import os
import sys
import pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location('crfmod', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_crf_final.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    train = pd.read_csv(m.TRAIN_CSV, keep_default_na=False)
    gaz = m.build_gazetteer(train, min_count=1)

    # restrict to high-confidence gazetteer: single type, count >= N
    def make_hc_gaz(min_count):
        hc = {}
        for expr, c in gaz.items():
            total = sum(c.values())
            if total >= min_count and len(c) == 1:
                hc[expr] = c.most_common(1)[0][0]
        return hc

    # Build validation set
    X_full = []
    y_full = []
    for idx, row in train.iterrows():
        sent = row['sentence']
        ents = m.parse_entities(row['entities'])
        tags = m.bio_tag(sent, ents)
        X_full.append(m.sent_features(sent, gaz))
        y_full.append(tags)

    from sklearn.model_selection import train_test_split
    idxs = list(range(len(train)))
    tr_idx, va_idx = train_test_split(idxs, test_size=0.15, random_state=42)
    X_tr = [X_full[i] for i in tr_idx]
    y_tr = [y_full[i] for i in tr_idx]
    X_va = [X_full[i] for i in va_idx]
    y_va = [y_full[i] for i in va_idx]
    va_sents = [train.iloc[i]['sentence'] for i in va_idx]
    va_gold = [train.iloc[i]['entities'] for i in va_idx]

    import sklearn_crfsuite
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.5, c2=0.15, max_iterations=80, all_possible_transitions=True)
    crf.fit(X_tr, y_tr)
    y_va_pred = crf.predict(X_va)
    base_pred = [m.entities_to_str(m.decode_bio(s, t)) for s, t in zip(va_sents, y_va_pred)]
    f1, p, r, tp, fp, fn = m.entity_f1(va_gold, base_pred)
    print(f'Base: F1={f1:.4f} P={p:.4f} R={r:.4f}', flush=True)

    def postprocess(sent, tags, hc_gaz):
        # current predicted entities as (start,end,type)
        cur = []
        i = 0
        n = len(sent)
        while i < n:
            t = tags[i]
            if t.startswith('B-'):
                typ = t[2:]
                start = i
                i += 1
                while i < n and tags[i] == 'I-' + typ:
                    i += 1
                cur.append((start, i, typ))
            else:
                i += 1
        covered = [False] * n
        for (s, e, typ) in cur:
            for j in range(s, e):
                covered[j] = True
        # add hc gazetteer matches not overlapping
        add = []
        for expr, typ in hc_gaz.items():
            start = 0
            while True:
                idx = sent.find(expr, start)
                if idx == -1:
                    break
                if not any(covered[idx:idx + len(expr)]):
                    add.append((idx, idx + len(expr), typ, expr))
                    for j in range(idx, idx + len(expr)):
                        covered[j] = True
                start = idx + len(expr)
        # merge
        all_ents = [(s, e, typ) for (s, e, typ) in cur]
        for (s, e, typ, expr) in add:
            all_ents.append((s, e, typ))
        all_ents.sort()
        ents = [(sent[s:e], typ) for (s, e, typ) in all_ents]
        return m.entities_to_str(ents)

    for min_count in [2, 3, 5, 8]:
        hc = make_hc_gaz(min_count)
        pp = [postprocess(s, t, hc) for s, t in zip(va_sents, y_va_pred)]
        f1, p, r, tp, fp, fn = m.entity_f1(va_gold, pp)
        print(f'PP min_count={min_count} (hc size={len(hc)}): F1={f1:.4f} P={p:.4f} R={r:.4f} TP={tp} FP={fp} FN={fn}', flush=True)


if __name__ == '__main__':
    main()
