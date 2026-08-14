"""Fast heuristic baseline: right-branching chain to root with position priors."""
from collections import Counter, defaultdict
from common import load_train, load_test, write_submission, las


def train_priors(rows):
    # label distribution by token position-from-end, and for the root
    pos_lab = defaultdict(Counter)
    root_lab = Counter()
    for r in rows:
        n = len(r['tokens'])
        for i, l in enumerate(r['labels']):
            d = n - (i + 1)  # distance from end (0 = last)
            pos_lab[min(d, 3)][l] += 1
        root_lab[r['labels'][-1]] += 1
    return pos_lab, root_lab


def predict(rows, pos_lab, root_lab):
    out = []
    for r in rows:
        n = len(r['tokens'])
        heads, labels = [], []
        for i in range(n):
            d = n - (i + 1)
            if d == 0:
                heads.append(0)
                labels.append(root_lab.most_common(1)[0][0])
            else:
                heads.append(i + 2)  # next token
                labels.append(pos_lab[min(d, 3)].most_common(1)[0][0])
        out.append({'id': r['id'], 'heads': heads, 'labels': labels})
    return out


if __name__ == '__main__':
    train = load_train()
    pos_lab, root_lab = train_priors(train)
    # self-eval on train (upper bound of this heuristic)
    print('train LAS (heuristic):', las(train, predict(train, pos_lab, root_lab)))
    test = load_test()
    write_submission(predict(test, pos_lab, root_lab), 'outputs/submission.csv')
    print('wrote outputs/submission.csv')
