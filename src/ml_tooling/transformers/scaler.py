import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class DFStandardScaler(BaseEstimator, TransformerMixin):
    """
    Wrapping of the StandardScaler from scikit-learn for Pandas DataFrames. See:
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = self.scaler.transform(X)
        return pd.DataFrame(data=data, columns=X.columns, index=X.index)