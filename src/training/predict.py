import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def predict(model_name, X, y):
    model = joblib.load(model_name)
    preds = model.predict(X)
    print(f'Accuracy of the model is {(preds == y).sum() / len(y_test)}')

TARGET = 'survived'
SEED_MODEL = 42


URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
df = pd.read_csv(URL)
X_train, X_test, y_train, y_test = train_test_split( df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=SEED_MODEL)
predict('models/titanic_20220305193145.sav', X_test, y_test)
print(df.columns)
print(df.iloc[0])