"""Shared helpers: metric, data loading, temporal hold-out split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error (the competition metric)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


def load_csv(data_dir, name):
    path = Path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle 'bike-sharing-demand' data "
            "(see README) or run: python src/make_synthetic.py")
    return pd.read_csv(path)


def holdout_mask(df, day=15):
    """True for rows on day-of-month >= `day` (validation), False for training.

    The real test set is the *later* days of each month, so this hold-out mimics it
    far better than random K-fold (which leaks neighbouring hours).
    """
    return (pd.to_datetime(df["datetime"]).dt.day >= day).to_numpy()
