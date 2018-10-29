"""
Transformers for use in sklearn Pipelines.
Mainly deals with DataFrames
"""
from typing import Union, Callable

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from functools import reduce

from .utils import listify, TransformerError


# noinspection PyUnusedLocal
class Select(BaseEstimator, TransformerMixin):
    """
    Selects columns from DataFrame
    """

    def __init__(self, columns):
        self.columns = listify(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise TransformerError(f"The DataFrame does not include the columns: {cols_error}")


# noinspection PyUnusedLocal
class FillNA(BaseEstimator, TransformerMixin):
    """
    Fills NA values with given value or strategy. Either a value or a strategy has to be supplied.
    """

    def __init__(self, value: int = None, strategy: str = None):
        self.value = value
        self.strategy = strategy
        self.column_values_ = None

        def _most_freq(X):
            return pd.DataFrame.mode(X).iloc[0]

        self.func_map_ = {'mean': pd.DataFrame.mean,
                          'median': pd.DataFrame.median,
                          'most_freq': _most_freq,
                          'max': pd.DataFrame.max,
                          'min': pd.DataFrame.min}

    def fit(self, X: pd.DataFrame, y=None):
        if self.value is None and self.strategy is None:
            raise TransformerError(f"Both value and strategy are set to None."
                                   f"Please select either a value or a strategy.")
        if self.value is not None and self.strategy is not None:
            raise TransformerError(f"Both a value and a strategy have been selected."
                                   f"Please select either a value or a strategy.")
        if self.strategy is not None:
            func = self.func_map_[self.strategy]
            self.column_values_ = func(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        if self.strategy is not None:
            result = X.fillna(self.column_values_)
        else:
            result = X.fillna(self.value)
        return result


# noinspection PyUnusedLocal
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
        X = X.copy()

        for col in X.columns:
            X[col] = pd.Categorical(X[col],
                                    categories=self.cat_map_[col].categories,
                                    ordered=self.cat_map_[col].ordered)
        return pd.get_dummies(X)


# noinspection PyUnusedLocal
class FuncTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a given function to each column
    """

    def __init__(self, func: Callable[[Union[pd.DataFrame, pd.Series]], pd.DataFrame]):
        self.func = func

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = self.func(X[col])
        return X


# noinspection PyUnusedLocal
class DFFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Merges together two pipelines based on index
    """

    def __init__(self, transformer_list: list):
        self.transformer_list = transformer_list

    def fit(self, X: pd.DataFrame, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)

        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


# noinspection PyUnusedLocal
class Binner(BaseEstimator, TransformerMixin):
    """
    Bins data according to passed bins and labels
    """

    def __init__(self, bins: list = None, labels: list = None):
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = pd.cut(X[col], bins=self.bins, labels=self.labels)
        return X


# noinspection PyUnusedLocal
class Renamer(BaseEstimator, TransformerMixin):
    """
    Renames columns to passed names
    """

    def __init__(self, column_names: Union[list, str]):
        self.column_names = listify(column_names)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if len(self.column_names) != len(X.columns):
            raise TransformerError(f"X has {len(X.columns)} columns - "
                                   f"You provided {len(self.column_names)} column names")
        X.columns = self.column_names
        return X


# noinspection PyUnusedLocal
class DateEncoder(BaseEstimator, TransformerMixin):
    """
    Converts a date column into multiple day-month-year columns
    """

    def __init__(self, day=True, month=True, week=True, year=True):
        self.day = day
        self.month = month
        self.week = week
        self.year = year

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if self.day:
                X[f"{col}_day"] = X[col].dt.day
            if self.month:
                X[f"{col}_month"] = X[col].dt.month
            if self.year:
                X[f"{col}_year"] = X[col].dt.year
            if self.week:
                X[f"{col}_week"] = X[col].dt.week
            X = X.drop(col, axis=1)
        return X


# noinspection PyUnusedLocal
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

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].str.upper().map(self.frequencies[col]).fillna(0)
        return X


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


class DFRowFunc(BaseEstimator, TransformerMixin):
    """
    Row-wise operation on Pandas DataFrame. Strategy can either be one of
    the predefined or a callable.    If some elements in the row are NaN these
    elements are ignored for the built-in strategies.
    """

    _func_map = {'sum': np.sum,
                 'min': np.min,
                 'max': np.max}

    def __init__(self, strategy=None):

        if strategy is None:
            raise TransformerError("No strategy is specified.")

        if not isinstance(strategy, str) and not callable(strategy):
            raise TransformerError(f"{strategy} is not a callable or a string.")

        if isinstance(strategy, str) and strategy not in self.func_map_.keys():
            raise TransformerError(f"Strategy {strategy} is not a predefined strategy.")

        if isinstance(strategy, str):
            self.func = self.func_map_[strategy]
        else:
            self.func = strategy

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.DataFrame(X.apply(self.func, axis=1))
        return X
