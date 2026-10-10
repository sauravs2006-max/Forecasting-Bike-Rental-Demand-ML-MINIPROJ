"""Load the saved model, predict on data/test.csv, write submission.csv (Kaggle format)."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_features
from utils import load_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--out", default="results/submission.csv")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    test = load_csv(a.data_dir, "test.csv")
    pred = np.clip(bundle["model"].predict(build_features(test, bundle["cfg"])), 0, None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"datetime": test["datetime"], "count": pred}).to_csv(a.out, index=False)
    print(f"Model: {bundle['name']} | wrote {len(pred)} predictions to {a.out}")


if __name__ == "__main__":
    main()
