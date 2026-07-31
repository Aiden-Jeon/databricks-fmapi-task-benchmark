"""Re-run pass B (out-of-fold scoring of train examples) after the fold fix."""
import os, pickle
import numpy as np
import run as R


def main():
    NB = int(os.environ.get("NB", 4000))
    tr, te = R.load_data()
    idf = R.init_idf(tr, te)
    ssim = R.init_ssim(tr, te)
    folds = np.load(os.path.join(R.CACHE, "folds.npy"))
    with open(os.path.join(R.CACHE, "stage1_models.pkl"), "rb") as f:
        models = pickle.load(f)
    p = os.path.join(R.CACHE, "stage2_rows.pkl")
    if os.path.exists(p):
        os.rename(p, p + ".leaked")
    recs = R.pass_b(tr.iloc[:NB].reset_index(drop=True), idf, models, folds[:NB], ssim,
                    nproc=int(os.environ.get("NPROC", 4)))
    f1 = np.array([r["f1"] for r in recs])
    un = np.array([r["unans"] for r in recs])
    print("mean f1 top1 %.4f | answerable-only %.4f | oracle top8 %.4f | unans %.3f"
          % (f1.mean(), f1[un == 0].mean(),
             np.array([r["f1_top"] for r in recs])[un == 0].mean(), un.mean()))


if __name__ == "__main__":
    main()
