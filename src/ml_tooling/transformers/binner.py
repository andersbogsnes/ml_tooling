from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Binner(BaseEstimator, TransformerMixin):
    """
    Bins data according to passed bins and labels. Uses pd.cut() under the hood,
    see https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html#pandas-cut
    for further details
    """

    def __init__(self, bins: Union[int, list], labels: list = None):
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = pd.cut(x_[col], bins=self.bins, labels=self.labels)
        return x_
