# Data

Place the Kaggle **Bike Sharing Demand** files here:

    data/train.csv
    data/test.csv

Download (needs a Kaggle account, accept the competition rules once on the website):

    pip install kaggle
    kaggle competitions download -c bike-sharing-demand -p data
    unzip data/bike-sharing-demand.zip -d data

Or download manually from https://www.kaggle.com/c/bike-sharing-demand/data

The CSV files are git-ignored. `python src/make_synthetic.py` creates a fake dataset with the same schema, for smoke tests only.
