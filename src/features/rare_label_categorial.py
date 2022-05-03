from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd


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
