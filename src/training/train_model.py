
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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED_MODEL = 42
NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']
TARGET = 'survived'

numeric_transformer = Pipeline(steps=[
    ('missing_indicator', MissingIndicator(NUMERICAL_VARS)),
    ('median_imputation', NumericalImputesEncoder(NUMERICAL_VARS)),

])
categorical_transformer = Pipeline(steps=[
    ('cabin_only_letter', CabinOnlyLetter('cabin')),
    ('categorical_imputer', CategoricalImputerEncoder(CATEGORICAL_VARS)),
    ('rare_labels', RareLabelCategoricalEncoder(tol=0.02,  variables=CATEGORICAL_VARS)),
    ('dummy_vars', OneHotEncoder(CATEGORICAL_VARS)),
])
column_transformer = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERICAL_VARS),
        ('cat', categorical_transformer, CATEGORICAL_VARS)],
    remainder="drop"
)
preprocessor = Pipeline(
    [        
        ('cleaning', CleaningTransformer()),
        ('column_transformer', column_transformer),
        ('scaling', MinMaxScaler()),
    ]
)
titanic_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('log_reg', LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL))
    ]
)
URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
df = pd.read_csv(URL)

X_train, X_test, y_train, y_test = train_test_split( df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=SEED_MODEL)

titanic_pipeline.fit(X_train, y_train)

preds = titanic_pipeline.predict(X_test)

print(f'Accuracy of the model is {(preds == y_test).sum() / len(y_test)}')


now = datetime.now()
date_time = now.strftime("%Y%d%m%H%M%S")

filename = f'titanic_{date_time}.sav'
print(f'Model stored in models as {filename}')
joblib.dump(titanic_pipeline, f"models/{filename}")