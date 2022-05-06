from operator import index
import pandas as pd
from sklearn.model_selection import train_test_split
from titanic_pipeline.config import config

df = pd.read_csv(config.URL).drop(columns="home.dest")
train, prod = train_test_split(df, test_size=0.5, random_state=42)

train.to_csv(config.DATASETS_DIR + "data.csv", index=False)
prod_data = prod.drop(config.TARGET, axis=1)
prod_truth = prod[[config.TARGET]]

prod_data.to_csv(config.DATASETS_DIR + "prod_data.csv", index=False)
prod_truth.to_csv(config.DATASETS_DIR + "prod_truth.csv", index=False)
