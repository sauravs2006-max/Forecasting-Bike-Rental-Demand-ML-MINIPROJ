"""Train, tune and compare all models; stack; save the best one.

Protocol
  * Tune every model with K-fold CV on the "early days" part of the training data
    (day-of-month < --holdout-day).
  * Report CV RMSLE and RMSLE on the later-days hold-out (mimics the real test split).
  * Refit the winner on ALL training data and save to models/best_model.joblib.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold

from features import FeatureConfig, build_features
from models import LogStack, get_models
from utils import holdout_mask, load_csv, rmsle, rmsle_scorer

STACK_BASES = ["Decision Tree", "Random Forest", "GBM (Hist)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--config", default=None, help="feature config JSON (e.g. results/best_config.json)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds (paper used 10)")
    ap.add_argument("--holdout-day", type=int, default=15)
    ap.add_argument("--fast", action="store_true", help="no grid search; quick smoke test")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out, fig_dir = Path(a.out), Path(a.out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True); Path(a.model_dir).mkdir(exist_ok=True)

    cfg = FeatureConfig.from_json(a.config) if a.config else FeatureConfig()
    print("Feature config:", cfg)
    df = load_csv(a.data_dir, "train.csv")
    X, y = build_features(df, cfg), df["count"].to_numpy()
    val = holdout_mask(df, a.holdout_day)
    Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]
    print(f"train part: {len(Xtr)} rows | hold-out: {len(Xva)} rows | features: {list(X.columns)}\n")

    kf = KFold(a.cv, shuffle=True, random_state=a.seed)
    rows, tuned, preds = [], {}, {}
    for name, (pipe, grid) in get_models(Xtr, a.fast, a.seed).items():
        t = time.time()
        gs = GridSearchCV(pipe, grid, cv=kf, scoring=rmsle_scorer, n_jobs=a.jobs, refit=True).fit(Xtr, ytr)
        p = gs.predict(Xva)
        tuned[name], preds[name] = gs.best_estimator_, p
        rows.append(dict(model=name, cv_rmsle=-gs.best_score_, holdout_rmsle=rmsle(yva, p),
                         best_params=json.dumps(gs.best_params_, default=str)))
        print(f"{name:15s} CV={rows[-1]['cv_rmsle']:.4f}  holdout={rows[-1]['holdout_rmsle']:.4f}  ({time.time()-t:.0f}s)")

    stack = LogStack([(n, tuned[n]) for n in STACK_BASES], cv=5, random_state=a.seed).fit(Xtr, ytr)
    preds["Stack"] = stack.predict(Xva)
    rows.append(dict(model="Stack", cv_rmsle=np.nan, holdout_rmsle=rmsle(yva, preds["Stack"]),
                     best_params="tree+RF+GBM -> linear meta (log space)"))
    print(f"{'Stack':15s} holdout={rows[-1]['holdout_rmsle']:.4f}")

    res = pd.DataFrame(rows).sort_values("holdout_rmsle").reset_index(drop=True)
    res.to_csv(out / "model_results.csv", index=False)
    (out / "results_table.md").write_text(
        res[["model", "cv_rmsle", "holdout_rmsle"]].round(4).to_markdown(index=False)
        if hasattr(res, "to_markdown") and _has_tabulate() else
        res[["model", "cv_rmsle", "holdout_rmsle"]].round(4).to_string(index=False))
    best = res.loc[0, "model"]
    print(f"\nBest model on hold-out: {best}")

    # refit the winner on all training data
    final = clone(stack if best == "Stack" else tuned[best]).fit(X, y)
    joblib.dump({"model": final, "cfg": cfg, "name": best}, Path(a.model_dir) / "best_model.joblib")
    (out / "best_model.txt").write_text(best)

    # ---------- figures ----------
    plt.figure(figsize=(8, 4))
    r = res.iloc[::-1]
    plt.barh(r["model"], r["holdout_rmsle"], color=["tab:red" if m == best else "tab:blue" for m in r["model"]])
    plt.xlabel("hold-out RMSLE (lower is better)"); plt.title("Model comparison")
    plt.tight_layout(); plt.savefig(fig_dir / "model_comparison.png", dpi=150); plt.close()

    hours = Xva["hour"].to_numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(pd.Series(yva).groupby(hours).mean(), marker="o", label="actual")
    plt.plot(pd.Series(preds[best]).groupby(hours).mean(), marker="x", label=f"predicted ({best})")
    plt.xlabel("hour"); plt.ylabel("mean rentals"); plt.title("Hold-out: actual vs predicted by hour")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(fig_dir / "pred_vs_actual.png", dpi=150); plt.close()

    rf = tuned["Random Forest"]
    pi = permutation_importance(rf, Xva, yva, scoring=rmsle_scorer, n_repeats=5, random_state=a.seed, n_jobs=a.jobs)
    # scorer is negative RMSLE, so importances_mean = (score before) - (score after shuffling) = RMSLE increase (>0 = important)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
    plt.figure(figsize=(6.5, 4)); imp.plot(kind="barh")
    plt.xlabel("RMSLE increase when shuffled"); plt.title("Permutation importance (Random Forest)")
    plt.tight_layout(); plt.savefig(fig_dir / "feature_importance.png", dpi=150); plt.close()
    imp.sort_values(ascending=False).to_csv(out / "feature_importance.csv", header=["importance"])
    print("Saved results to", out, "and model to", a.model_dir)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
