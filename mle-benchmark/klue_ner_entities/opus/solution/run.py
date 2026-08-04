import os, sys, time, argparse, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ner import (TYPES, Decoder, LABELS, NL, parse_ents, ents_to_spans, spans_to_labels,
                 labels_to_spans, spans_to_str, Gazetteer, sent_features,
                 FeatureMap, Perceptron, micro_f1)

T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


def encode_set(sents, gaz, fm, freeze, quiet=False):
    fm.frozen = freeze
    data = []
    for k, s in enumerate(sents):
        gm = gaz.match(s) if gaz is not None else None
        feats = sent_features(s, gm)
        idx, cnts = fm.encode(feats)
        offs = np.zeros(len(cnts), dtype=np.int64)
        np.cumsum(cnts[:-1], out=offs[1:])
        data.append((idx, offs, len(s)))
        if not quiet and (k + 1) % 4000 == 0:
            log('  encoded %d/%d (vocab=%d)' % (k + 1, len(sents), len(fm.d)))
    return data


def encode_train_kfold(sents, spans, fm, k=5, seed=13):
    """Encode training sentences using out-of-fold gazetteers so that the model
    does not learn to blindly trust the (in-sample-perfect) gazetteer."""
    n = len(sents)
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, k, size=n)
    data = [None] * n
    for f in range(k):
        oth = np.nonzero(fold != f)[0]
        cur = np.nonzero(fold == f)[0]
        g = Gazetteer().fit([sents[i] for i in oth], [spans[i] for i in oth])
        sub = encode_set([sents[i] for i in cur], g, fm, freeze=False, quiet=True)
        for j, i in enumerate(cur):
            data[i] = sub[j]
        log('  fold %d/%d done (gaz=%d, vocab=%d)' % (f + 1, k, len(g.d), len(fm.d)))
    return data


def train_one(data, golds, nfeat, epochs, seed=0, dev=None, dev_sents=None,
              dev_gold_strs=None, tag=''):
    model = Perceptron(nfeat)
    rng = np.random.RandomState(seed)
    order = np.arange(len(data))
    for ep in range(epochs):
        rng.shuffle(order)
        err = tot = 0
        for i in order:
            idx, offs, n = data[i]
            pred = model.decode(idx, offs, n)
            g = golds[i]
            if not np.array_equal(pred, g):
                err += model.update(idx, offs, g, pred)
            tot += n
        msg = '%sepoch %d  tokerr=%.4f' % (tag, ep + 1, err / tot)
        if dev is not None and (ep + 1) % 2 == 0:
            model.use_avg()
            preds = [spans_to_str(s, labels_to_spans(model.decode(*d)))
                     for s, d in zip(dev_sents, dev)]
            pr, rc, f = micro_f1(dev_gold_strs, preds)
            msg += '  dev P=%.4f R=%.4f F1=%.4f' % (pr, rc, f)
            model.use_raw()
        log(msg)
    W, T = model.averaged()
    del model
    return W, T


def train(data, golds, nfeat, epochs, seeds=(0,), **kw):
    Wsum = Tsum = None
    for k, sd in enumerate(seeds):
        W, T = train_one(data, golds, nfeat, epochs, seed=sd,
                         tag='m%d ' % k, **kw)
        if Wsum is None:
            Wsum, Tsum = W, T
        else:
            Wsum += W
            Tsum += T
            del W, T
        if len(seeds) > 1 and kw.get('dev') is not None:
            m = Decoder(Wsum / (k + 1), Tsum / (k + 1))
            preds = [spans_to_str(s, labels_to_spans(m.decode(*d)))
                     for s, d in zip(kw['dev_sents'], kw['dev'])]
            pr, rc, f = micro_f1(kw['dev_gold_strs'], preds)
            log('ENSEMBLE(%d) dev P=%.4f R=%.4f F1=%.4f' % (k + 1, pr, rc, f))
            del m
    n = len(seeds)
    if n > 1:
        Wsum /= n
        Tsum /= n
    return Decoder(Wsum, Tsum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='full', choices=['dev', 'full'])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--biases', default='0')
    ap.add_argument('--obias', type=float, default=0.0)
    ap.add_argument('--out', default='outputs/submission.csv')
    args = ap.parse_args()

    tr = pd.read_csv('train.csv', keep_default_na=False)
    te = pd.read_csv('test.csv', keep_default_na=False)
    if args.limit:
        tr = tr.iloc[:args.limit].reset_index(drop=True)

    tr_sents = tr['sentence'].tolist()
    tr_spans = [ents_to_spans(s, parse_ents(e))
                for s, e in zip(tr_sents, tr['entities'])]

    if args.mode == 'dev':
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(tr_sents))
        ncut = int(len(perm) * 0.85)
        tri, dvi = perm[:ncut], perm[ncut:]
        fit_sents = [tr_sents[i] for i in tri]
        fit_spans = [tr_spans[i] for i in tri]
        ev_sents = [tr_sents[i] for i in dvi]
        ev_gold = [tr['entities'].iloc[i] for i in dvi]
        ev_ids = None
    else:
        fit_sents, fit_spans = tr_sents, tr_spans
        ev_sents = te['sentence'].tolist()
        ev_gold = None
        ev_ids = te['id'].tolist()

    log('fit=%d eval=%d' % (len(fit_sents), len(ev_sents)))
    gaz = Gazetteer().fit(fit_sents, fit_spans)
    log('gazetteer size=%d' % len(gaz.d))

    fm = FeatureMap()
    if args.kfold > 1:
        Xtr = encode_train_kfold(fit_sents, fit_spans, fm, k=args.kfold)
    else:
        Xtr = encode_set(fit_sents, gaz, fm, freeze=False)
    log('train encoded, vocab=%d' % len(fm.d))
    Xev = encode_set(ev_sents, gaz, fm, freeze=True)
    log('eval encoded')

    golds = [spans_to_labels(len(s), sp) for s, sp in zip(fit_sents, fit_spans)]
    nfeat = len(fm.d)

    dev_args = {}
    if ev_gold is not None:
        dev_args = dict(dev=Xev, dev_sents=ev_sents, dev_gold_strs=ev_gold)
    model = train(Xtr, golds, nfeat, args.epochs,
                  seeds=tuple(range(args.seeds)), **dev_args)

    biases = [float(x) for x in args.biases.split(',')]
    for b in biases:
        preds = [spans_to_str(s, labels_to_spans(model.decode(*d, o_bias=b)))
                 for s, d in zip(ev_sents, Xev)]
        if ev_gold is not None:
            p, r, f = micro_f1(ev_gold, preds)
            log('FINAL bias=%.2f dev P=%.4f R=%.4f F1=%.4f' % (b, p, r, f))
        else:
            out = args.out if b == args.obias else (
                'outputs/cand_b%g.csv' % b)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pd.DataFrame({'id': ev_ids, 'entities': preds}).to_csv(
                out, index=False)
            log('wrote %s (bias=%g)' % (out, b))


if __name__ == '__main__':
    main()
