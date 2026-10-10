#!/usr/bin/env bash
# End-to-end pipeline. Usage: bash run_all.sh [--fast]
set -e
FAST=$1
python src/eda.py
python src/ablation.py
python src/train.py --config results/best_config.json --cv 10 $FAST
python src/predict.py
echo "Done. See results/ (tables, figures, submission.csv)."
