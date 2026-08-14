"""Train structured perceptron on full train split, eval on dev."""
import sys, os, time, random, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perceptron import *
from memm2 import extract_feats, build_gold_tags, make_gazetteer
from common import *


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--beam', type=int, default=20)
    ap.add_argument('--dev', type=int, default=2000)
    ap.add_argument('--evaln', type=int, default=1000)
    ap.add_argument('--minlen', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save', type=str, default='')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--tag', type=str, default='p')
    args = ap.parse_args()

    rows = load_csv(f"{TASK_DIR}/train.csv")
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if args.full:
        train_rows = rows
        dev_rows = None
    else:
        train_rows = [rows[i] for i in idx[args.dev:]]
        dev_rows = [rows[i] for i in idx[:args.dev]]

    gaz_map, gaz_maxlen = make_gazetteer(train_rows)
    print(f"[{args.tag}] gaz {len(gaz_map)}", flush=True)

    t0 = time.time()
    train_feats = [extract_feats(r['sentence'], gaz_map, gaz_maxlen) for r in train_rows]
    train_tags = build_gold_tags(train_rows)
    print(f"[{args.tag}] feats {time.time()-t0:.0f}s ({len(train_feats)} sents)", flush=True)

    per = train(train_feats, train_tags, epochs=args.epochs, beam=args.beam,
                seed=args.seed, min_ent_len=args.minlen)
    if args.save:
        save(per, args.save)
        print(f"[{args.tag}] saved {args.save}", flush=True)

    if dev_rows is not None:
        eval_rows = dev_rows[:args.evaln]
        dev_sents = [r['sentence'] for r in eval_rows]
        t0 = time.time()
        dev_feats = [extract_feats(s, gaz_map, gaz_maxlen) for s in dev_sents]
        gold = {r['id']: parse_entities(r['entities']) for r in eval_rows}
        pred = {}
        for i, r in enumerate(eval_rows):
            path = decode(per, dev_feats[i], beam=args.beam, min_ent_len=args.minlen)
            pred[r['id']] = tags_to_spans(dev_sents[i], path)
        p, rc, f1 = score_f1(gold, pred)
        print(f"[{args.tag}] DEV F1 {f1:.4f} P {p:.4f} R {rc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        # per-type
        from collections import defaultdict as dd, Counter
        pert = dd(lambda: [0, 0, 0])
        for k in gold:
            gc = Counter(gold[k]); pc = Counter(pred[k])
            for key in set(gc) | set(pc):
                t = key[1]
                pert[t][0] += min(gc[key], pc[key])
                pert[t][1] += max(0, pc[key] - gc[key])
                pert[t][2] += max(0, gc[key] - pc[key])
        for t in LABELS:
            a, b, c = pert[t]
            pp = a / (a + b) if a + b else 0
            rr = a / (a + c) if a + c else 0
            ff = 2 * pp * rr / (pp + rr + 1e-9)
            print(f"  {t}: P {pp:.3f} R {rr:.3f} F1 {ff:.3f}", flush=True)

    if args.out:
        test_rows = load_csv(f"{TASK_DIR}/test.csv")
        t0 = time.time()
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(['id', 'entities'])
            for row in test_rows:
                spans = predict_sent(per, row['sentence'], gaz_map, gaz_maxlen,
                                     beam=args.beam, min_ent_len=args.minlen)
                w.writerow([row['id'], entities_to_str(spans)])
        print(f"[{args.tag}] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
