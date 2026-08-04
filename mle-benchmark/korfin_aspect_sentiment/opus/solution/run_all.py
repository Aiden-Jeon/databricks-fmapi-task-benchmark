"""End-to-end reproducible pipeline for t23_korfin_asc.

Usage:  python solution/run_all.py [--fresh]

Stages
  1. oof.py     - 17 base models on TF-IDF views of the aspect-masked sentence
  2. views.py   -  9 extra base models on directional / clause / josa-stripped views
  3. blend.py   - greedy blend of base OOF probabilities (diagnostic)
  4. sibpred.py - leak-free sentence-group-out-of-fold predictions
  5. final.py   - LGBM/LR stack over base probs + target-encoding features,
                  multi-seed averaging -> cache/FINAL_test.npy
  6. make_submission.py FINAL -> outputs/submission.csv
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
STEPS = ["oof.py", "views.py", "blend.py", "sibpred.py", "final.py"]

if "--fresh" in sys.argv and os.path.isdir(CACHE):
    shutil.rmtree(CACHE)

for s in STEPS:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)

print("\n=== make_submission.py ===", flush=True)
subprocess.run([sys.executable, os.path.join(HERE, "make_submission.py"), "FINAL"],
               check=True)
