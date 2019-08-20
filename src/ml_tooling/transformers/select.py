from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_tooling.utils import listify, TransformerError


class Select(BaseEstimator, TransformerMixin):
    """
    Selects columns from DataFrame
    """

    def __init__(self, columns: Union[list, str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = listify(self.columns)
        return self

    def transform(self, X: pd.DataFrame):
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise TransformerError(
                f"The DataFrame does not include the columns: {cols_error}"
            )
