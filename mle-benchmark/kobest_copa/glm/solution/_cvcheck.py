import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solution as S
import pandas as pd

tr = pd.read_csv('../train.csv'); tr['question'] = tr['question'].str.strip()
S.N_SEEDS = 2
accs = S.cross_validate(tr, n_splits=5)
print('CV (2 seeds):', round(accs.mean(), 4), accs.tolist(), flush=True)
print('DONE', flush=True)
