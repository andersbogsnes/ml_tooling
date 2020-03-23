from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_tooling.utils import TransformerError


class FuncTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a given function to each column
    """

    def __init__(self, func: Callable[..., pd.DataFrame], **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = self.func(x_[col], **self.kwargs)
        return x_


class DFRowFunc(BaseEstimator, TransformerMixin):
    """Row-wise operation on Pandas DataFrame. Strategy can either be one of
    the predefined or a callable. If some elements in the row are NaN these
    elements are ignored for the built-in strategies.
    Valid strategies are:

        - sum
        - min
        - max
        - mean

    If a callable is used, it must return a pd.Series
    """

    _func_map = {"sum": np.sum, "min": np.min, "max": np.max, "mean": np.mean}

    def __init__(self, strategy=None):
        self.strategy = strategy
        self.func = None

    # noinspection PyUnusedLocal
    def fit(self, X: pd.DataFrame, y=None):
        self._validate_strategy(self.strategy)
        return self

    def _validate_strategy(self, strategy):
        if strategy is None:
            raise TransformerError("No strategy is specified.")

        if not isinstance(strategy, str) and not callable(strategy):
            raise TransformerError(f"{strategy} is not a callable or a string.")

        if isinstance(strategy, str) and strategy not in self._func_map:
            raise TransformerError(f"Strategy {strategy} is not a predefined strategy.")

        if isinstance(strategy, str):
            self.func = self._func_map[strategy]
        else:
            self.func = strategy

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        if self.strategy in self._func_map:
            x_ = self._func_map.get(self.strategy)(x_, axis=1).to_frame()
        else:
            x_ = x_.apply(self.func, axis=1).to_frame()
        return x_
