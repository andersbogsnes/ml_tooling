import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FreqFeature(BaseEstimator, TransformerMixin):
    """
    Converts a column into its normalized value count
    """

    def __init__(self):
        self.frequencies = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.columns:
            self.frequencies[col] = X[col].str.upper().value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = x_[col].str.upper().map(self.frequencies[col]).fillna(0)
        return x_
