
import joblib
from sklearn.compose import ColumnTransformer

from src.training.transformers import (
    MissingIndicator, CabinOnlyLetter, CategoricalImputerEncoder,
    NumericalImputesEncoder, RareLabelCategoricalEncoder,
    OneHotEncoder, MinMaxScaler, CleaningTransformer
)
import pandas as pd
import numpy as np 
from datetime import datetime

from src import config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split
from pathlib import Path

numeric_transformer = Pipeline(steps=[
    ('missing_indicator', MissingIndicator(config.NUMERICAL_VARS)),
    ('median_imputation', NumericalImputesEncoder(config.NUMERICAL_VARS)),

])
categorical_transformer = Pipeline(steps=[
    ('cabin_only_letter', CabinOnlyLetter('cabin')),
    ('categorical_imputer', CategoricalImputerEncoder(config.CATEGORICAL_VARS)),
    ('rare_labels', RareLabelCategoricalEncoder(tol=0.02,  variables=config.CATEGORICAL_VARS)),
    ('dummy_vars', OneHotEncoder(config.CATEGORICAL_VARS)),
])
column_transformer = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, config.NUMERICAL_VARS),
        ('cat', categorical_transformer, config.CATEGORICAL_VARS)],
    remainder="drop"
)
preprocessor = Pipeline(
    [        
        ('cleaning', CleaningTransformer()),
        ('column_transformer', column_transformer),
        ('scaling', MinMaxScaler()),
    ]
)
if config.MODEL_NAME == 'RandomForest':
    regressor = RandomForestClassifier(max_depth=4, class_weight='balanced', random_state=config.SEED_MODEL)
else:
    regressor = LogisticRegression(C=0.0005, class_weight='balanced', random_state=config.SEED_MODEL)

titanic_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ]
)
df = pd.read_csv(config.DATASET_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(config.TARGET, axis=1), df[config.TARGET], test_size=0.2,
    random_state=config.SEED_MODEL
)
titanic_pipeline.fit(X_train, y_train)
preds = titanic_pipeline.predict(X_test)
print(preds)
print(f'Accuracy of the model is {(preds == y_test).sum() / len(y_test)}')

now = datetime.now()
date_time = now.strftime("%Y%d%m%H%M%S")

filename = f'{config.MODEL_NAME}_{date_time}'
print(f'Model stored in models as {filename}')
joblib.dump(titanic_pipeline, f"models/{filename}.sav")

# p = Path(config.DATASETS_DIR, filename)
# p.mkdir(exist_ok=True)
# joblib.dump(X_train, f"{p}/X_train.csv")
# joblib.dump(X_test, f"{p}/X_test.csv")
# joblib.dump(y_train, f"{p}/y_train.csv")
# joblib.dump(y_test, f"{p}/y_test.csv")

