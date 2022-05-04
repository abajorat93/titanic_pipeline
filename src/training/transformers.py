from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as oneHot
from sklearn.preprocessing import MinMaxScaler as MMScaler


from typing import List
import pandas as pd
import re
import numpy as np


def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
# Keep only one cabin

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
    
class CleaningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df: pd.DataFrame,y=0):
        return self

    def transform(self, df: pd.DataFrame,y=0):
        df.replace('?', np.nan, inplace=True)
        df['age'] = df['age'].astype('float')
        df['fare'] = df['fare'].astype('float')
        df['cabin'] = df['cabin'].apply(get_first_cabin)
        df['title'] = df['name'].apply(get_title)
        return df

class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol, variables: List[str]):
        self.tol = tol
        self.variables = variables
        self.valid_labels_dict = {}

    def fit(self, dat_df: pd.DataFrame,y=0):
        for var in self.variables:
            t = dat_df[var].value_counts() / dat_df.shape[0]
            self.valid_labels_dict[var] = t[t > self.tol].index.tolist()
        return self

    def transform(self, data_df: pd.DataFrame,y=0):
        for var in self.variables:
            tmp = [col for col in data_df[var].unique() if col not in self.valid_labels_dict[var]]
            data_df[var] = data_df[var].replace(to_replace=tmp, value=len(tmp) * ['Rare'])
        return data_df

class CabinOnlyLetter(BaseEstimator, TransformerMixin):

    def __init__(self, column: str):
        self.column = column

    def fit(self, x: pd.DataFrame,y=0):
        return self

    def transform(self, X: pd.DataFrame,y=0):

        X[self.column] = [''.join(re.findall("[a-zA-Z]+", row)) if type(row) == str else row for row in X[self.column]]

        return X
    
class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables: List[str]):
        self.encoder = oneHot(handle_unknown = 'ignore',drop='first')
        self.variables = variables

    def fit(self, X:pd.DataFrame,y=0) -> None:
        self.encoder.fit(X[self.variables])
        return self

    def transform(self, X:pd.DataFrame,y=0) -> None:
        X[self.encoder.get_feature_names_out(self.variables)] = self.encoder.transform(X[self.variables]).toarray()
        X.drop(self.variables, axis=1, inplace=True)
        return X

class MissingIndicator(BaseEstimator, TransformerMixin):

  def __init__(self, columnsList: List[str]):
    self.columnsList = columnsList

  def fit(self, x: pd.DataFrame, y=0):
    return self

  def transform(self, X: pd.DataFrame,y=0):
    for column in self.columnsList:
      X[f"{column}_nan"] = X[column].isnull().astype(int)
    return X

class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = MMScaler()

    def fit(self, X:pd.DataFrame,y=0):
        self.scaler.fit(X)
        return self

    def transform(self, X:pd.DataFrame,y=0):
        X_scaled = self.scaler.transform(X)
        return X_scaled

class NumericalImputesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables
        self.valid_labels_dict = {}

    def fit(self, data_df: pd.DataFrame,y=0):
        for var in self.variables:
            t = data_df[var].median()
            self.valid_labels_dict[var] = t
        return self

    def transform(self, data_df: pd.DataFrame,y=0):
        for var in self.variables:
            data_df[var] = data_df[var].fillna(self.valid_labels_dict[var])
        return data_df
    
class CategoricalImputerEncoder(BaseEstimator, TransformerMixin):
    """Función que reemplaza los valores nulos de una columna

    Args:
        BaseEstimator (BaseEstimator): Clase heredada
        TransformerMixin (TransformerMixin): Clase heredada
    """

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame,y=0):
        return self

    def transform(self, X: pd.DataFrame,y=0):
        X[self.variables] = X[self.variables].fillna("missing")
        return X
    