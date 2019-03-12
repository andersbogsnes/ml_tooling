import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ToCategorical(BaseEstimator, TransformerMixin):
    """
    Converts a column into a one-hot encoded column through pd.Categorical
    """

    def __init__(self):
        self.cat_map_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_map_ = {col: X[col].astype('category').cat for
                         col in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            x_[col] = pd.Categorical(x_[col],
                                     categories=self.cat_map_[col].categories,
                                     ordered=self.cat_map_[col].ordered)
        return pd.get_dummies(x_)


class Binarize(BaseEstimator, TransformerMixin):
    """
    Sets all instances of value to 1 and all others to 0
    Returns a pandas DataFrame
    """
    def __init__(self, value):
        self.value = value

    # noinspection PyUnusedLocal
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        data = np.where(x_ == self.value, 1, 0)
        return pd.DataFrame(data=data, columns=x_.columns, index=x_.index)