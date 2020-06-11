import pandas as pd
from typing import Any, Union
from ml_tooling.metrics.utils import _is_percent
from sklearn.base import BaseEstimator, TransformerMixin


class RareFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: Union[int, float] = 0.2, fill_rare: Any = "Rare"):
        """
        Replaces categories with a specified value, if they occur less often than
        the provided threshold.

        Parameters
        ----------
        threshold: int, float
            Sets the threshold for when a value is considered rare.
            Any value which occurs less than the threshold will be replaced
            with fill_rare. If threshold is a float, it will be considered
            a percentage and if it is an int, threshold will be considered the
            minimum number of observations.
        fill_rare: Any
            Fill value to use when replacing rare categories.
        """
        self.threshold = threshold
        self.fill_rare = fill_rare
        self.rare_values_dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.columns:
            frequencies = X[col].value_counts(normalize=_is_percent(self.threshold))

            rare_values = frequencies[frequencies < self.threshold].index
            self.rare_values_dict[col] = dict.fromkeys(rare_values, self.fill_rare)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            if col in self.rare_values_dict:
                x_[col] = x_[col].replace(self.rare_values_dict[col])

        return x_
