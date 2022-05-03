import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

filename = 'src/models/finalized_model.sav'

loaded_model = joblib.load(filename)

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

TARGET = 'survived'
SEED_MODEL = 42

X_train, X_test, y_train, y_test = train_test_split( df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=SEED_MODEL)

result = loaded_model.score(X_test, y_test)
print(result)