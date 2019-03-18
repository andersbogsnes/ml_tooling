from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_tooling.utils import listify, TransformerError


class Renamer(BaseEstimator, TransformerMixin):
    """
    Renames columns to passed names.
    """

    def __init__(self, column_names: Union[list, str]):
        self.column_names = column_names

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()

        column_names = listify(self.column_names)

        if len(column_names) != len(x_.columns):
            raise TransformerError(f"X has {len(x_.columns)} columns - "
                                   f"You provided {len(column_names)} column names")
        x_.columns = column_names
        return x_
