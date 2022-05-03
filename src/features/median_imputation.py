from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd


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
