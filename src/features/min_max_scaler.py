from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler as MMScaler
import pandas as pd 

class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = MMScaler()

    def fit(self, X:pd.DataFrame,y=0):
        self.scaler.fit(X)
        return self

    def transform(self, X:pd.DataFrame,y=0):
        X_scaled = self.scaler.transform(X)
        return X_scaled