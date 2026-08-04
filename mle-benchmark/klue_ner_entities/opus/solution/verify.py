import sys, pandas as pd
from collections import Counter
sys.path.insert(0, 'solution')
from ner import parse_ents, TYPES

path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/submission.csv'
sub = pd.read_csv(path, keep_default_na=False)
te = pd.read_csv('test.csv', keep_default_na=False)
ss = pd.read_csv('sample_submission.csv', keep_default_na=False)

assert list(sub.columns) == list(ss.columns), (sub.columns, ss.columns)
assert len(sub) == len(te), (len(sub), len(te))
assert Counter(sub['id']) == Counter(te['id']), 'id mismatch'
assert list(sub['id']) == list(te['id']), 'id order differs (ok but noting)'

sent = dict(zip(te['id'], te['sentence']))
bad = 0
nent = 0
tyc = Counter()
for i, e in zip(sub['id'], sub['entities']):
    ents = parse_ents(e)
    nent += len(ents)
    s = sent[i]
    pos = 0
    for t, ty in ents:
        tyc[ty] += 1
        if ty not in TYPES:
            bad += 1
        j = s.find(t, pos)
        if j < 0:
            bad += 1
        else:
            pos = j + len(t)
assert bad == 0, 'bad entities: %d' % bad
print('OK rows=%d entities=%d (%.2f/sent) empty=%d' %
      (len(sub), nent, nent / len(sub), sum(1 for x in sub['entities'] if not x)))
print('types', tyc)
