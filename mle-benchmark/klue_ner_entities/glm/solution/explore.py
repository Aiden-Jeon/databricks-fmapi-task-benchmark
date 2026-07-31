"""Explore entity-to-character mapping to design BIO tagging."""
import pandas as pd
from collections import Counter

train = pd.read_csv('train.csv', keep_default_na=False)

# Count how many sentences have a duplicate entity string appearing multiple times
dup_entity_count = 0
ambig = 0
for idx, row in train.iterrows():
    sent = row['sentence']
    ents = row['entities']
    if not ents:
        continue
    pairs = [p for p in ents.split('|') if p]
    # count duplicate expressions within the entity list
    exprs = [p.rsplit(':', 1)[0] for p in pairs]
    expr_counts = Counter(exprs)
    if any(v > 1 for v in expr_counts.values()):
        dup_entity_count += 1
    # Check for ambiguity: an expression appearing in sentence more times than in entity list
    for e, c in expr_counts.items():
        n_in_sent = sent.count(e)
        if n_in_sent != c:
            # Could be overlapping matches
            pass

print('Sentences with duplicate entity expressions in list:', dup_entity_count)

# Greedy left-to-right matching check: how many entities can be matched unambiguously
def greedy_match(sent, exprs):
    tags = ['O'] * len(sent)
    used = [False] * len(sent)
    pos = 0
    for expr, typ in exprs:
        # find next occurrence starting from pos
        idx = sent.find(expr, pos)
        if idx == -1:
            return None
        for j in range(idx, idx + len(expr)):
            if used[j]:
                return None
            used[j] = True
        tags[idx] = 'B-' + typ
        for j in range(idx + 1, idx + len(expr)):
            tags[j] = 'I-' + typ
        pos = idx + len(expr)
    return tags

fail = 0
for idx, row in train.iterrows():
    sent = row['sentence']
    ents = row['entities']
    if not ents:
        continue
    pairs = [p for p in ents.split('|') if p]
    exprs = [(p.rsplit(':', 1)[0], p.rsplit(':', 1)[1]) for p in pairs]
    if greedy_match(sent, exprs) is None:
        fail += 1
        if fail <= 5:
            print('GREEDY FAIL:', repr(sent[:80]), ents)
print('Greedy match failures:', fail)
