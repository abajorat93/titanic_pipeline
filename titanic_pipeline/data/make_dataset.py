from operator import index
import pandas as pd
from sklearn.model_selection import train_test_split
from titanic_pipeline.config import config

df = pd.read_csv(config.URL)
train, prod = train_test_split(df, test_size=0.5, random_state=42)
prod1, prod2 = train_test_split(prod, test_size=0.5, random_state=42)

train.to_csv(config.DATASETS_DIR + 'data.csv', index=False)
prod1_data = prod1.drop(config.TARGET, axis=1)
prod1_truth = prod1[[config.TARGET]]

prod2_data = prod2.drop(config.TARGET, axis=1)
prod2_truth = prod2[[config.TARGET]]

prod1_data.to_csv(config.DATASETS_DIR + 'prod1_data.csv', index=False)
prod2_data.to_csv(config.DATASETS_DIR + 'prod2_data.csv', index=False)

prod1_truth.to_csv(config.DATASETS_DIR + 'prod1_truth.csv', index=False)
prod2_truth.to_csv(config.DATASETS_DIR + 'prod2_truth.csv', index=False)

