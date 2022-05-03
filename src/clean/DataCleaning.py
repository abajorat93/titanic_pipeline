import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class GetFirstCabinEncoder(BaseEstimator, TransformerMixin):
    """Función que obtiene el primer valor de la cabina en
    el caso de que exista.

    Args:
        BaseEstimator (BaseEstimator): Clase heredada
        TransformerMixin (TransformerMixin): Clase heredada
    """

    def __init__(self):
        return None

    def fit(self):
        return self  # pass

    def transform(self, df: pd.DataFrame):
        for row in df.itertuples():
            idx = row.Index
            try:
                df.at[idx, "cabin"] = df.at[idx, "cabin"].split()[0]
            except ValueError:
                df.at[idx, "cabin"] = np.nan
        return df


class GetTitleEncoder(BaseEstimator, TransformerMixin):
    """Función que obtiene el título de la persona.
    Args:
        BaseEstimator (BaseEstimator): Clase heredada
        TransformerMixin (TransformerMixin): Clase heredada
    """

    def __init__(self):
        return None

    def fit(self):
        return None  # pass | self

    def transform(self, df: pd.DataFrame):
        for row in df.itertuples():
            line = df.at[row.Index, "name"]
            if re.search("Mrs", line):
                df.at[row.Index, "title"] = "Mrs"
            elif re.search("Mr", line):
                df.at[row.Index, "title"] = "Mr"
            elif re.search("Miss", line):
                df.at[row.Index, "title"] = "Miss"
            elif re.search("Master", line):
                df.at[row.Index, "title"] = "Master"
            else:
                df.at[row.Index, "title"] = "Other"
        return df
