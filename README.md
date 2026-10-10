# Forecasting Bike Rental Demand

UE24CS352A Machine Learning mini-project by Greeshma K (PES1UG24AM368) and Saurav S (PES1UG24AM398).

The goal is to predict how many bikes get rented in the next hour, given the date, time and weather. We use the Capital Bikeshare data from Washington D.C. (the Kaggle Bike Sharing Demand dataset) and are scored on RMSLE, which cares about percentage error rather than the raw number of bikes.

## What we did

First we looked at the data (src/eda.py): hourly patterns, seasons, the effect of temperature, and the fact that the rental counts are quite skewed.

Then we tried six feature ideas from the reference paper (src/features.py): month instead of season, day of week, dropping temp, dropping holiday, rush-hour flags, and bucketing the weather values. src/ablation.py tries all 64 combinations and keeps the best one.

For models (src/models.py) we used linear regression, Poisson GLM, PCR, SVR, gradient boosting, random forest and a decision tree, plus a stacked model on top. They are trained on log1p(count) so that plain squared error matches RMSLE.

We checked them in two ways (src/train.py): normal K-fold cross-validation, and a time-based hold-out where we train on days 1-14 of each month and validate on days 15-19. The hold-out is closer to the real test set, which is days 20 onwards. After grid-search tuning, the best model is refit on all the training data.

Finally, src/predict.py writes results/submission.csv, and src/demo.py predicts a single hour for the live demo.

A few things differ from the paper. scikit-learn has no conditional inference tree, so a tuned DecisionTreeRegressor stands in for it. PoissonRegressor only does L2, with no elastic-net. HistGradientBoosting replaces R's gbm. We also always add year as a feature, since demand clearly grew in 2012.

## Setup

```bash
git clone https://github.com/sauravs2006-max/Forecasting-Bike-Rental-Demand-ML-MINIPROJ.git
cd Forecasting-Bike-Rental-Demand-ML-MINIPROJ
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put train.csv and test.csv in the data/ folder (see data/README.md for where to get them). If scikit-learn won't run on your machine, Google Colab works too: upload the project, unzip it, install the requirements, and run the same commands with a ! in front.

## Running it

```bash
python src/eda.py
python src/ablation.py
python src/train.py --config results/best_config.json --cv 10
python src/predict.py
python src/demo.py --datetime "2012-06-15 08:00:00" --temp 24 --humidity 55 --windspeed 10 --weather 1
```

Or run everything with `bash run_all.sh` (add --fast for a quick test without grid search). If it takes too long, use --cv 5 instead of --cv 10; the results barely change.

To generate the write-up PDF and the slides from the results:

```bash
pip install -r requirements-report.txt
python src/make_report.py --team "Greeshma K (PES1UG24AM368), Saurav S (PES1UG24AM398)" --problem "Forecasting Bike Rental Demand" --repo https://github.com/sauravs2006-max/Forecasting-Bike-Rental-Demand-ML-MINIPROJ
python src/make_slides.py --team "Greeshma K (PES1UG24AM368), Saurav S (PES1UG24AM398)" --problem "Forecasting Bike Rental Demand"
```

Both scripts read their numbers from results/ and refuse to run on the synthetic practice data.

## Outputs

Everything generated ends up in results/: model_results.csv and results_table.md (CV and hold-out RMSLE for every model), ablation.csv and best_config.json (the 64 feature combinations and the chosen one), feature_importance.csv, figures/ (EDA and model plots) and submission.csv (Kaggle-format predictions). The trained model is saved to models/best_model.joblib, which is not tracked in git.

The code is in src/, the write-up and slide material in docs/, and the Kaggle CSVs go in data/ (also not tracked).
