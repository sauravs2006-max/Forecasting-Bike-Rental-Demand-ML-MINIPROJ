"""Exploratory plots -> results/figures/eda_*.png"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    fig_dir = Path(a.out) / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(a.data_dir, "train.csv")
    dt = pd.to_datetime(df["datetime"])
    df["hour"], df["month"], df["dow"] = dt.dt.hour, dt.dt.month, dt.dt.dayofweek
    print(df.describe().T.to_string())
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("\nseason values seen per month (sanity check):\n",
          df.groupby("month")["season"].unique().to_string())

    # 1. hourly profile by working day
    plt.figure(figsize=(7, 4))
    for wd, lab in [(1, "working day"), (0, "weekend/holiday")]:
        plt.plot(df[df.workingday == wd].groupby("hour")["count"].mean(), marker="o", label=lab)
    plt.xlabel("hour of day"); plt.ylabel("mean rentals"); plt.title("Hourly demand profile")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(fig_dir / "eda_hourly.png", dpi=150); plt.close()

    # 2. demand vs temperature
    plt.figure(figsize=(7, 4))
    plt.scatter(df["atemp"], df["count"], s=4, alpha=.25)
    plt.xlabel("atemp"); plt.ylabel("count"); plt.title("Demand vs feels-like temperature")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_temp.png", dpi=150); plt.close()

    # 3. monthly mean
    plt.figure(figsize=(7, 4))
    df.groupby("month")["count"].mean().plot(kind="bar")
    plt.ylabel("mean rentals"); plt.title("Seasonality"); plt.tight_layout()
    plt.savefig(fig_dir / "eda_month.png", dpi=150); plt.close()

    # 4. target distribution (raw vs log) -> why RMSLE / log target
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(df["count"], bins=50); ax[0].set_title("count (right-skewed)")
    ax[1].hist(np.log1p(df["count"]), bins=50); ax[1].set_title("log1p(count)")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_target.png", dpi=150); plt.close()

    # 5. correlation heatmap
    num = df.select_dtypes("number").drop(columns=["casual", "registered"])
    c = num.corr()
    plt.figure(figsize=(7, 6)); plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(c)), c.columns, rotation=90); plt.yticks(range(len(c)), c.columns)
    plt.colorbar(); plt.title("Correlation matrix"); plt.tight_layout()
    plt.savefig(fig_dir / "eda_corr.png", dpi=150); plt.close()
    print(f"\nSaved EDA figures to {fig_dir}")


if __name__ == "__main__":
    main()
