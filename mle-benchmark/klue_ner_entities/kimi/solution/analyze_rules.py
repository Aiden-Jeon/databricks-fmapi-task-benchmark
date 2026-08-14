import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from collections import Counter

rows = load_csv(f"{TASK_DIR}/train.csv")
fps = Counter(); fns = Counter()
for row in rows[:3000]:
    sent = row['sentence']
    gold = parse_entities(row['entities'])
    gold_spans = set()
    for e, t in gold:
        start = 0
        while True:
            i = sent.find(e, start)
            if i < 0: break
            gold_spans.add((i, i+len(e), t))
            start = i+1
    pred = set(extract_rule_spans(sent))
    for s,e,t in pred - gold_spans:
        fps[(sent[s:e], t)] += 1
    for s,e,t in gold_spans - pred:
        if t in ('DT','TI','QT'):
            fns[(sent[s:e], t)] += 1

print("=== top rule FPs ===")
for (e,t),c in fps.most_common(40): print(c, t, repr(e))
print("\n=== top DT/TI/QT FNs (not covered) ===")
for (e,t),c in fns.most_common(40): print(c, t, repr(e))
