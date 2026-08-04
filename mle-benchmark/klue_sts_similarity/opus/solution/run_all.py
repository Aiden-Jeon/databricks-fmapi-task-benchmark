"""End-to-end reproducible pipeline for t7_klue_sts.

Usage (from the task root directory):
    python solution/run_all.py

Stages
------
1. cv.build      -> dense feature block A  (solution/_cache.npz)
2. build2        -> dense feature block B  (solution/_cache2.npz)
3. sparse_model  -> sparse pair ridge, char+word    (solution/_sparse.npz)
4. sparse_model2 -> sparse pair ridge, jamo+wordbi  (solution/_sparse2.npz)
5. eval2         -> 5-fold OOF/test preds for dense base models
6. final_blend   -> NNLS blend -> outputs/submission.csv

Everything is trained on train.csv only. Vectorizers / SVD / KMeans are
unsupervised and fitted on the sentence corpus (train + test sentences, no
labels), which is a transductive but label-free step.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STAGES = [
    ("dense features A", [sys.executable, os.path.join(HERE, "cv.py")]),
    ("dense features B", [sys.executable, os.path.join(HERE, "build2.py")]),
    ("sparse pair ridge (char+word)", [sys.executable, os.path.join(HERE, "sparse_model.py")]),
    ("sparse pair ridge (jamo+wordbi)", [sys.executable, os.path.join(HERE, "sparse_model2.py")]),
    ("base models: ridge svr hgb hgb2", [sys.executable, os.path.join(HERE, "eval2.py"),
                                         "ridge", "svr", "hgb", "hgb2"]),
    ("base models: et", [sys.executable, os.path.join(HERE, "eval2.py"), "et"]),
    ("base models: mlp mlp2 hgb_abs", [sys.executable, os.path.join(HERE, "eval2.py"),
                                       "mlp", "mlp2", "hgb_abs"]),
    ("stacked hgb (dense + sparse OOF)", [sys.executable, os.path.join(HERE, "stack_try.py")]),
    ("stacked hgb/svr variants", [sys.executable, os.path.join(HERE, "stack_try2.py")]),
    ("final blend + submission", [sys.executable, os.path.join(HERE, "final_blend.py")]),
]


def main():
    t0 = time.time()
    for name, cmd in STAGES:
        print(f"\n===== {name} =====", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"----- done ({time.time()-t0:.0f}s elapsed)", flush=True)
    print("\nSubmission at outputs/submission.csv")


if __name__ == "__main__":
    main()
