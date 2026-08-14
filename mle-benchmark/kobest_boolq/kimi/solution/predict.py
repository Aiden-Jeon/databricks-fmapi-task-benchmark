"""Load artifacts produced by train.py, predict labels for test.csv, and write
outputs/submission.csv with every test id exactly once."""
import pickle
from pathlib import Path

import pandas as pd

from features import build_matrix

ROOT = Path(__file__).resolve().parent.parent


def main():
    test_df = pd.read_csv(ROOT / "test.csv")

    with open(ROOT / "solution" / "model.pkl", "rb") as f:
        art = pickle.load(f)
    v_word, v_diff, v_q = art["vectorizers"]
    model = art["model"]

    X_test, _ = build_matrix(test_df, v_word, v_diff, v_q)
    preds = model.predict(X_test).astype(int)

    sub = pd.DataFrame({"id": test_df["id"], "label": preds})
    # keep original test id order, every id exactly once
    assert sub["id"].is_unique
    assert set(sub["id"]) == set(test_df["id"])
    assert len(sub) == len(test_df)
    (ROOT / "outputs").mkdir(exist_ok=True)
    sub.to_csv(ROOT / "outputs" / "submission.csv", index=False)
    print(f"wrote outputs/submission.csv ({len(sub)} rows), "
          f"model={art['model_name']}, label dist={sub['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
