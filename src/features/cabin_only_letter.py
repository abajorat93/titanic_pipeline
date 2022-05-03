import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class CabinOnlyLetter(BaseEstimator, TransformerMixin):

    def __init__(self, column: str):
        self.column = column

    def fit(self, x: pd.DataFrame,y=0):
        return self

    def transform(self, X: pd.DataFrame,y=0):

        X[self.column] = [''.join(re.findall("[a-zA-Z]+", row)) if type(row) == str else row for row in X[self.column]]

        return X



