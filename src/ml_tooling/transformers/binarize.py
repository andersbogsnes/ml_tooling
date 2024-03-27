"""
Implements a Binner Transformer which will convert selected value to 1's and all other
values to 0
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Binarize(BaseEstimator, TransformerMixin):
    """
    Sets all instances of value to 1 and all others to 0
    Returns a pandas DataFrame
    """

    def __init__(self, value: Any = None):
        """
        Parameters
        ----------
        value: Any
            The value to be set to 1
        """
        self.value = value

    # noinspection PyUnusedLocal
    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_ = x.copy()
        data = np.where(x_ == self.value, 1, 0)
        return pd.DataFrame(data=data, columns=x_.columns, index=x_.index)
