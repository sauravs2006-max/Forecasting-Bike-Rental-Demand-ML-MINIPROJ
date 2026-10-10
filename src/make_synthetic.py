"""Generate a synthetic dataset with the SAME schema as Kaggle's bike-sharing-demand.

Only for smoke-testing the pipeline when the real data isn't available.
Train = days 1-19 of each month, test = days 20+ (like the real split).
Never report results from synthetic data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    idx = pd.date_range("2011-01-01", "2012-12-31 23:00", freq="h")
    df = pd.DataFrame({"datetime": idx})
    m, h, dow = idx.month, idx.hour, idx.dayofweek
    df["season"] = ((m - 1) // 3 + 1)
    df["holiday"] = 0
    df["workingday"] = (dow < 5).astype(int)
    df["weather"] = rng.choice([1, 2, 3, 4], size=len(df), p=[.66, .26, .079, .001])
    base_t = 16 - 12 * np.cos((idx.dayofyear - 15) / 365 * 2 * np.pi) + 5 * np.sin((h - 9) / 24 * 2 * np.pi)
    df["temp"] = np.round(base_t + rng.normal(0, 2, len(df)), 2)
    df["atemp"] = np.round(df["temp"] + rng.normal(1, 1.5, len(df)), 2)
    df["humidity"] = np.clip(rng.normal(62, 19, len(df)), 0, 100).round().astype(int)
    df["windspeed"] = np.clip(rng.gamma(2.5, 5, len(df)), 0, 56).round(4)

    wk = np.where(np.isin(h, [8, 17, 18]), 5.0, np.where(np.isin(h, [7, 9, 16, 19]), 3.0,
                  np.where((h >= 10) & (h <= 15), 1.6, np.where(h <= 5, 0.15, 0.8))))
    we = np.where((h >= 10) & (h <= 18), 3.2, np.where(h <= 6, 0.3, 1.0))
    shape = np.where(df["workingday"] == 1, wk, we)
    temp_eff = np.clip((df["temp"] + 5) / 35, 0.05, 1.3)
    weather_eff = df["weather"].map({1: 1.0, 2: .85, 3: .5, 4: .2}).to_numpy()
    growth = np.where(idx.year == 2012, 1.7, 1.0)
    lam = 45 * shape * temp_eff * weather_eff * growth
    cnt = rng.poisson(lam).clip(1)
    df["casual"] = (cnt * rng.uniform(.1, .35, len(df))).astype(int)
    df["registered"] = cnt - df["casual"]
    df["count"] = cnt

    is_train = idx.day <= 19
    out = Path(a.out); out.mkdir(exist_ok=True)
    tr = df[is_train].copy()
    te = df[~is_train].drop(columns=["casual", "registered", "count"])
    for d in (tr, te):
        d["datetime"] = d["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tr.to_csv(out / "train.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    print(f"wrote {out/'train.csv'} ({len(tr)} rows) and {out/'test.csv'} ({len(te)} rows)")


if __name__ == "__main__":
    main()
