import pandas as pd
from typing import Any, Union
from ml_tooling.metrics.utils import _is_percent
from sklearn.base import BaseEstimator, TransformerMixin


class RareFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces categories with a value, if they occur less than a threshold.
    """

    def __init__(self, threshold: Union[int, float] = 0.2, fill_rare: Any = "Rare"):
        self.threshold = threshold
        self.fill_rare = fill_rare
        self.mask_dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.columns:
            normalize = _is_percent(self.threshold)
            frequencies = X[col].value_counts(normalize=normalize)

            mask_obs = frequencies[frequencies < self.threshold].index
            self.mask_dict[col] = dict.fromkeys(mask_obs, self.fill_rare)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            if col in self.mask_dict:
                x_[col] = x_[col].replace(self.mask_dict[col])

        return x_
