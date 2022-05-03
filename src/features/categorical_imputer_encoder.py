import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


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