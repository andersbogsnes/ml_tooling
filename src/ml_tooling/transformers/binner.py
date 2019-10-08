from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Binner(BaseEstimator, TransformerMixin):
    """
    Bins data according to passed bins and labels. Uses :meth:`pandas.cut()` under the hood,
    see for further details
    """

    def __init__(self, bins: Union[int, list] = 5, labels: list = None):
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = pd.cut(x_[col], bins=self.bins, labels=self.labels)
        return x_
