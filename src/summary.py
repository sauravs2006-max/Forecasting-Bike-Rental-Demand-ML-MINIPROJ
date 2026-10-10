"""Collect numbers from data/ and results/ so the write-up and slides are filled from REAL outputs.

Nothing here invents a number: every value comes from train.csv or from the files written by
eda.py / ablation.py / train.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_TRAIN_ROWS = 10886          # size of the real Kaggle train.csv
LINEAR_FAMILY = ["Linear", "Poisson GLM", "PCR"]
TREE_FAMILY = ["Decision Tree", "Random Forest", "GBM (Hist)", "Stack"]


def collect(data_dir="data", results_dir="results", allow_synthetic=False):
    data_dir, res = Path(data_dir), Path(results_dir)
    need = [res / "model_results.csv", res / "ablation.csv", res / "best_config.json",
            res / "feature_importance.csv", data_dir / "train.csv"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit("Missing files (run the pipeline first, see README): " + ", ".join(missing))

    df = pd.read_csv(data_dir / "train.csv")
    if len(df) != KAGGLE_TRAIN_ROWS and not allow_synthetic:
        raise SystemExit(
            f"data/train.csv has {len(df)} rows, but the real Kaggle file has {KAGGLE_TRAIN_ROWS}. "
            "You may be using the synthetic data - never report results from it. "
            "(Use --allow-synthetic only for a layout test.)")

    dt = pd.to_datetime(df["datetime"])
    df = df.assign(hour=dt.dt.hour, year=dt.dt.year)
    f = {"n_train": len(df), "n_missing": int(df.isna().sum().sum()),
         "date_min": dt.min().strftime("%Y-%m-%d"), "date_max": dt.max().strftime("%Y-%m-%d"),
         "synthetic": len(df) != KAGGLE_TRAIN_ROWS}
    t = data_dir / "test.csv"
    f["n_test"] = len(pd.read_csv(t)) if t.exists() else None

    # --- EDA facts (computed, not assumed)
    wd = df[df.workingday == 1].groupby("hour")["count"].mean()
    we = df[df.workingday == 0].groupby("hour")["count"].mean()
    f["wd_peaks"] = sorted(wd.sort_values(ascending=False).index[:2].tolist())
    f["we_peak"] = int(we.idxmax())
    y = df.groupby("year")["count"].mean()
    f["year_means"] = {int(k): float(v) for k, v in y.items()}
    f["count_mean"], f["count_median"] = float(df["count"].mean()), float(df["count"].median())
    f["count_skew"] = float(df["count"].skew())
    f["corr_atemp"] = float(df["count"].corr(df["atemp"]))
    f["weather_means"] = {int(k): float(v) for k, v in df.groupby("weather")["count"].mean().items()}

    # --- models
    m = pd.read_csv(res / "model_results.csv").sort_values("holdout_rmsle").reset_index(drop=True)
    f["models"] = m
    f["best"], f["best_holdout"] = m.loc[0, "model"], float(m.loc[0, "holdout_rmsle"])
    f["best_cv"] = None if pd.isna(m.loc[0, "cv_rmsle"]) else float(m.loc[0, "cv_rmsle"])
    for key, fam in [("linear", LINEAR_FAMILY), ("tree", TREE_FAMILY)]:
        sub = m[m.model.isin(fam)]
        f[key + "_best"] = (sub.iloc[0]["model"], float(sub.iloc[0]["holdout_rmsle"])) if len(sub) else (None, None)

    # --- ablation / feature config
    ab = pd.read_csv(res / "ablation.csv")
    flags = [c for c in ab.columns if c not in ("cv_rmsle", "holdout_rmsle", "score")]
    f["n_subsets"] = len(ab)
    f["ab_best"], f["ab_worst"] = float(ab["score"].min()), float(ab["score"].max())
    base = ab[~ab[flags].astype(bool).any(axis=1)]
    f["ab_baseline"] = float(base["score"].iloc[0]) if len(base) else None
    f["cfg"] = json.loads((res / "best_config.json").read_text())
    f["cfg_on"] = [k for k, v in f["cfg"].items() if v]

    # --- importance
    imp = pd.read_csv(res / "feature_importance.csv", index_col=0).iloc[:, 0]
    f["top_features"] = imp.sort_values(ascending=False).head(3).index.tolist()
    return f


CFG_TEXT = {
    "month_for_season": "month instead of season", "add_dow": "day-of-week",
    "drop_temp": "temp dropped (atemp kept)", "drop_holiday": "holiday dropped",
    "peak_hours": "weekday/weekend peak-hour flags", "discretize": "discretised weather variables",
}


def cfg_sentence(f):
    on = [CFG_TEXT[k] for k in f["cfg_on"]]
    return ", ".join(on) if on else "no optional transformations (raw features + hour + year)"


def conclusions(f):
    """Plain-language conclusions that only state what the numbers support."""
    out = []
    cv = f"{f['best_cv']:.4f}" if f["best_cv"] is not None else "n/a for the stack"
    out.append(f"The best model, {f['best']}, reached a hold-out RMSLE of {f['best_holdout']:.4f} (CV RMSLE: {cv}).")
    ln, lv = f["linear_best"]; tn, tv = f["tree_best"]
    if ln and tn:
        if tv < lv:
            out.append(f"Tree-based/ensemble models ({tn}: {tv:.4f}) beat the best linear-family model ({ln}: {lv:.4f}), "
                       "consistent with a non-linear dependence of demand on hour, working day and weather.")
        else:
            out.append(f"The best linear-family model ({ln}: {lv:.4f}) was not beaten by the tree-based/ensemble models "
                       f"(best {tn}: {tv:.4f}) in our setup.")
    out.append("Most important features (permutation importance, Random Forest): " + ", ".join(f["top_features"]) + ".")
    if f["ab_baseline"] is not None:
        out.append(f"The {f['n_subsets']}-subset feature search lowered the mean (CV, hold-out) RMSLE from "
                   f"{f['ab_baseline']:.4f} (no optional features) to {f['ab_best']:.4f}.")
    return out


LIMITS = ("Limitations: the hold-out is days 15-19 of each month, a proxy for the real test split (days 20+); "
          "no lag/rolling demand features (this is not a true time-series model); casual and registered riders "
          "are not modelled separately; one random seed and a limited grid.")
FUTURE = ("Future work: lag and rolling features, cyclic hour encoding, separate casual/registered models, "
          "per-model feature sets, wider hyper-parameter search.")
