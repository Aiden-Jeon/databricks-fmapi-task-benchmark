import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from collections import Counter, defaultdict

rows = load_csv(f"{TASK_DIR}/train.csv")
print("n train:", len(rows))

type_cnt = Counter()
nent = 0
notfound = 0
amb = 0
expr_types = defaultdict(Counter)
lengths = []
for row in rows:
    ents = parse_entities(row['entities'])
    sent = row['sentence']
    for e, t in ents:
        nent += 1
        type_cnt[t] += 1
        expr_types[e][t] += 1
        lengths.append(len(e))
        if e not in sent:
            notfound += 1
            if notfound <= 10:
                print("NOT FOUND:", repr(e), "in", repr(sent[:100]))
for e, c in expr_types.items():
    if len(c) > 1:
        amb += 1
print("total ents:", nent, "type dist:", type_cnt)
print("not found in sentence:", notfound)
print("ambiguous exprs:", amb, "of", len(expr_types))
import statistics
print("expr len: mean", statistics.mean(lengths), "max", max(lengths))

# no-entity sentences
noent = sum(1 for r in rows if not parse_entities(r['entities']))
print("sentences w/o entities:", noent)

# test regex rules coverage on DT/TI/QT
from common import extract_rule_spans
tp = fp = fn = 0
per = defaultdict(lambda: [0,0,0])
for row in rows[:4000]:
    sent = row['sentence']
    gold = parse_entities(row['entities'])
    gold_spans = Counter()
    for e, t in gold:
        # find all occurrences
        start = 0
        idxs = []
        while True:
            i = sent.find(e, start)
            if i < 0: break
            idxs.append((i, i+len(e)))
            start = i+1
        for (s,en) in idxs:
            gold_spans[(s,en,t)] += 1
    pred = extract_rule_spans(sent)
    pred_spans = Counter((s,e,t) for s,e,t in pred)
    for k in set(gold_spans)|set(pred_spans):
        g,p = gold_spans[k], pred_spans[k]
        tp += min(g,p); fp += max(0,p-g); fn += max(0,g-p)
        if k[2] in ('DT','TI','QT'):
            per[k[2]][0]+=min(g,p); per[k[2]][1]+=max(0,p-g); per[k[2]][2]+=max(0,g-p)
prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
print("rule-only span-level (all types) P/R/F1 on 4000:", prec, rec, 2*prec*rec/(prec+rec+1e-9))
for t,(a,b,c) in per.items():
    pp=a/(a+b) if a+b else 0; rr=a/(a+c) if a+c else 0
    print(f"  {t}: P {pp:.3f} R {rr:.3f} (tp{a} fp{b} fn{c})")
