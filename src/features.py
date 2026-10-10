"""Feature engineering (the 6 ideas from the reference paper) as toggle-able flags."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["season", "weather", "hour", "month", "dow", "year"]
# fixed bin edges used when `discretize=True`
DISC_BINS = {
    "temp": [10, 20, 30],
    "atemp": [10, 20, 30],
    "humidity": [40, 60, 80],
    "windspeed": [7, 15, 25],
}
WEEKDAY_PEAK = [7, 8, 9, 17, 18, 19]
WEEKEND_PEAK = list(range(10, 19))


@dataclass(frozen=True)
class FeatureConfig:
    month_for_season: bool = False  # replace `season` by `month`
    add_dow: bool = False           # add day-of-week
    drop_temp: bool = False         # drop `temp`, keep `atemp`
    drop_holiday: bool = True       # drop `holiday`
    peak_hours: bool = True         # weekday / weekend peak-hour indicators
    discretize: bool = False        # bucket continuous weather variables

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path):
        return FeatureConfig(**json.loads(Path(path).read_text()))


def build_features(df, cfg=FeatureConfig()):
    dt = pd.to_datetime(df["datetime"])
    X = df.drop(columns=[c for c in ["datetime", "casual", "registered", "count"] if c in df]).copy()
    X["hour"] = dt.dt.hour
    X["year"] = dt.dt.year  # demand grew year over year
    if cfg.month_for_season:
        X["month"] = dt.dt.month
        X = X.drop(columns="season")
    if cfg.add_dow:
        X["dow"] = dt.dt.dayofweek
    if cfg.drop_temp:
        X = X.drop(columns="temp")
    if cfg.drop_holiday:
        X = X.drop(columns="holiday")
    if cfg.peak_hours:
        X["peak_weekday"] = ((X["workingday"] == 1) & X["hour"].isin(WEEKDAY_PEAK)).astype(int)
        X["peak_weekend"] = ((X["workingday"] == 0) & X["hour"].isin(WEEKEND_PEAK)).astype(int)
    if cfg.discretize:
        for col, edges in DISC_BINS.items():
            if col in X:
                X[col] = np.digitize(X[col], edges)
    return X
