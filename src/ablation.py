"""Try all 2^6 = 64 feature-engineering subsets (as in the paper) with a fast tree model.

Ranks by the mean of 5-fold CV RMSLE and temporal-holdout RMSLE, then writes
results/ablation.csv and results/best_config.json (used by train.py --config).
"""
import argparse
import itertools
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from features import FeatureConfig, build_features
from models import log_target
from utils import holdout_mask, load_csv, rmsle, rmsle_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_csv(a.data_dir, "train.csv")
    y = df["count"].to_numpy()
    val = holdout_mask(df)
    est = log_target(DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=a.seed))
    kf = KFold(a.folds, shuffle=True, random_state=a.seed)

    rows = []
    names = list(asdict(FeatureConfig()).keys())
    for flags in itertools.product([False, True], repeat=len(names)):
        cfg = FeatureConfig(**dict(zip(names, flags)))
        X = build_features(df, cfg)
        cv = -cross_val_score(clone(est), X, y, cv=kf, scoring=rmsle_scorer).mean()
        ho = rmsle(y[val], clone(est).fit(X[~val], y[~val]).predict(X[val]))
        rows.append({**asdict(cfg), "cv_rmsle": cv, "holdout_rmsle": ho, "score": (cv + ho) / 2})
    res = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    res.to_csv(out / "ablation.csv", index=False)

    best = res.iloc[0]
    FeatureConfig(**{k: bool(best[k]) for k in names}).to_json(out / "best_config.json")
    print(res.head(8).round(4).to_string())
    print("\nBest config saved to", out / "best_config.json")

    plt.figure(figsize=(9, 3.8))
    plt.bar(range(len(res)), res["score"]); plt.axhline(res["score"].iloc[0], color="r", lw=.8)
    plt.xlabel("feature subset (sorted)"); plt.ylabel("mean(CV, holdout) RMSLE")
    plt.title("64 feature-engineering subsets (best in red line)")
    plt.tight_layout(); plt.savefig(out / "figures" / "ablation.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
