"""Train MEMM on train split, tune o_bias on dev, report F1."""
import sys, os, time, random, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memm2 import *
from common import *


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--eta0', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--dev', type=int, default=2000)
    ap.add_argument('--evaln', type=int, default=800)
    ap.add_argument('--cache', type=str, default='')
    args = ap.parse_args()

    rows = load_csv(f"{TASK_DIR}/train.csv")
    rng = random.Random(42)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    train_rows = [rows[i] for i in idx[args.dev:]]
    dev_rows = [rows[i] for i in idx[:args.dev]]
    gaz_map, gaz_maxlen = make_gazetteer(train_rows)
    print(f"gaz {len(gaz_map)}", flush=True)

    cache_model = args.cache + '.model' if args.cache else ''
    cache_feats = args.cache + '.feats' if args.cache else ''

    if cache_model and os.path.exists(cache_model):
        model = pickle.load(open(cache_model, 'rb'))
        print("loaded model", flush=True)
    else:
        if cache_feats and os.path.exists(cache_feats):
            train_feats, train_tags = pickle.load(open(cache_feats, 'rb'))
            print("loaded feats", flush=True)
        else:
            t0 = time.time()
            train_feats = [extract_feats(r['sentence'], gaz_map, gaz_maxlen) for r in train_rows]
            train_tags = build_gold_tags(train_rows)
            print(f"feats {time.time()-t0:.0f}s", flush=True)
            if cache_feats:
                pickle.dump((train_feats, train_tags), open(cache_feats, 'wb'))
        t0 = time.time()
        model = MemmModel(alpha=args.alpha, eta0=args.eta0)
        model.fit(train_feats, train_tags, epochs=args.epochs)
        print(f"train {time.time()-t0:.0f}s", flush=True)
        if cache_model:
            pickle.dump(model, open(cache_model, 'wb'))

    eval_rows = dev_rows[:args.evaln]
    dev_sents = [r['sentence'] for r in eval_rows]
    t0 = time.time()
    dev_feats = [extract_feats(s, gaz_map, gaz_maxlen) for s in dev_sents]
    print(f"dev feats {time.time()-t0:.0f}s", flush=True)
    gold = {r['id']: parse_entities(r['entities']) for r in eval_rows}

    def eval_with(ob):
        pred = {}
        for i, r in enumerate(eval_rows):
            S = model.char_scores(dev_feats[i], o_bias=ob)
            path = beam_decode(S, beam=12)
            pred[r['id']] = tags_to_spans(dev_sents[i], path)
        return score_f1(gold, pred)

    best = None
    for ob in [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]:
        t0 = time.time()
        p, rc, f1 = eval_with(ob)
        print(f"o_bias={ob}: F1 {f1:.4f} P {p:.4f} R {rc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if best is None or f1 > best[1]:
            best = (ob, f1)
    print(f"BEST o_bias={best[0]} F1 {best[1]:.4f}", flush=True)


if __name__ == '__main__':
    main()
