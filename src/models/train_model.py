
import joblib
import sklearn
from src.features.missing_indicator import MissingIndicator
from src.features.cabin_only_letter import CabinOnlyLetter
from src.features.categorical_imputer_encoder import CategoricalImputerEncoder
from src.features.median_imputation import NumericalImputesEncoder
from src.features.rare_label_categorial import RareLabelCategoricalEncoder
from src.features.one_hot_encoder import OneHotEncoder
from src.features.min_max_scaler import MinMaxScaler



import pandas as pd
import re
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

SEED_MODEL = 42
NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']
TARGET = 'survived'

titanic_pipeline = Pipeline(
    [
        ('missing_indicator', MissingIndicator(NUMERICAL_VARS)),
        ('cabin_only_letter', CabinOnlyLetter('cabin')),
        ('categorical_imputer', CategoricalImputerEncoder(CATEGORICAL_VARS)),
        ('median_imputation', NumericalImputesEncoder(NUMERICAL_VARS)),
        ('rare_labels', RareLabelCategoricalEncoder(tol=0.02,  variables=CATEGORICAL_VARS)),
        ('dummy_vars', OneHotEncoder(CATEGORICAL_VARS)),
        ('scaling', MinMaxScaler()),
        ('log_reg', LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL))
        
    ]
)



#data = 'src/models/cleaned_data'
# Loading data from specific url
#df = pd.read_csv(data)

URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
# Loading data from specific url
df = pd.read_csv(URL)

# Uncovering missing data
df.replace('?', np.nan, inplace=True)
df['age'] = df['age'].astype('float')
df['fare'] = df['fare'].astype('float')

# helper function 1
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
# Keep only one cabin
df['cabin'] = df['cabin'].apply(get_first_cabin)

# helper function 2
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
# Extract the title from 'name'
df['title'] = df['name'].apply(get_title)

# Droping irrelevant columns
DROP_COLS = ['boat','body','home.dest','ticket','name']
df.drop(DROP_COLS, axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split( df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=SEED_MODEL)

titanic_pipeline.fit(X_train, y_train)

preds = titanic_pipeline.predict(X_test)

print(f'Accuracy of the model is {(preds == y_test).sum() / len(y_test)}')


filename = 'titanic_pipeline_model.sav'
joblib.dump(titanic_pipeline, filename)


# El archivo se corre con python -m src.models.train.model