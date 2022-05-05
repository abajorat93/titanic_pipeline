import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config

df = pd.read_csv(config.URL)
train, prod = train_test_split(df, test_size=0.5, random_state=42)
prod1, prod2 = train_test_split(prod, test_size=0.5, random_state=42)

train.to_csv(config.DATASETS_DIR + 'data.csv')
prod1.to_csv(config.DATASETS_DIR + 'prod1.csv')
prod2.to_csv(config.DATASETS_DIR + 'prod2.csv')
