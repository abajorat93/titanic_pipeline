import pytest
import pandas as pd
from src.features.rare_label_categorial import RareLabelCategoricalEncoder


def GetRareLabelTestData():
    df = pd.read_csv('rare_label_csv_test')
    result = pd.read_csv('rare_labels_csv_result')
    return [(df, result)]


@pytest.mark.parametrize("df, result", GetRareLabelTestData())
def test_extract_only_letter(df, result):
    cat_vars = ['sex', 'cabin', 'embarked', 'title']
    rare_labels = RareLabelCategoricalEncoder(tol=0.02, variables=cat_vars)
    rare_labels.fit(df)
    test_result = rare_labels.transform(df)
    assert test_result.equals(result)
