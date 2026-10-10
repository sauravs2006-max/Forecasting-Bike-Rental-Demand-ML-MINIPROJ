"""Interactive demo: predict demand for one hour typed on the command line.

Example:
  python src/demo.py --datetime "2012-06-15 08:00:00" --temp 24 --humidity 55 --windspeed 10 --weather 1 --workingday 1
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--datetime", required=True, help='e.g. "2012-06-15 08:00:00"')
    ap.add_argument("--temp", type=float, required=True, help="deg C")
    ap.add_argument("--atemp", type=float, default=None, help="feels-like deg C (default = temp)")
    ap.add_argument("--humidity", type=float, default=60)
    ap.add_argument("--windspeed", type=float, default=10)
    ap.add_argument("--weather", type=int, default=1, choices=[1, 2, 3, 4],
                    help="1 clear, 2 mist, 3 light rain/snow, 4 heavy rain/snow")
    ap.add_argument("--workingday", type=int, default=None, choices=[0, 1])
    ap.add_argument("--holiday", type=int, default=0, choices=[0, 1])
    a = ap.parse_args()

    dt = pd.Timestamp(a.datetime)
    row = pd.DataFrame([{
        "datetime": a.datetime,
        "season": (dt.month - 1) // 3 + 1,  # Kaggle: 1 = Jan-Mar, 2 = Apr-Jun, 3 = Jul-Sep, 4 = Oct-Dec
        "holiday": a.holiday,
        "workingday": int(dt.dayofweek < 5 and not a.holiday) if a.workingday is None else a.workingday,
        "weather": a.weather, "temp": a.temp, "atemp": a.atemp if a.atemp is not None else a.temp,
        "humidity": a.humidity, "windspeed": a.windspeed}])
    if not Path(a.model).exists():
        raise SystemExit(f"{a.model} not found - run `python src/train.py --config results/best_config.json` first.")
    b = joblib.load(a.model)
    print(f"Predicted rentals for {a.datetime}: {max(0.0, float(b['model'].predict(build_features(row, b['cfg']))[0])):.0f}")


if __name__ == "__main__":
    main()
