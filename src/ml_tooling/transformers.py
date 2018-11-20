"""
Transformers for use in sklearn Pipelines.
Mainly deals with DataFrames
"""
from typing import Union, Callable, Optional

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from functools import reduce

from .utils import listify, TransformerError, _most_freq, DataType


# noinspection PyUnusedLocal
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
            raise TransformerError(f"The DataFrame does not include the columns: {cols_error}")


# noinspection PyUnusedLocal
class FillNA(BaseEstimator, TransformerMixin):
    """
    Fills NA values with given value or strategy. Either a value or a strategy has to be supplied.
    """

    def __init__(self,
                 value: Optional[Union[str, int]] = None,
                 strategy: Optional[str] = None):

        self.value = value
        self.strategy = strategy
        self.value_map_ = None

        self.func_map_ = {'mean': pd.DataFrame.mean,
                          'median': pd.DataFrame.median,
                          'most_freq': _most_freq,
                          'max': pd.DataFrame.max,
                          'min': pd.DataFrame.min}

    def _validate_input(self):

        if self.value is None and self.strategy is None:
            raise TransformerError(f"Both value and strategy are set to None."
                                   f"Please select either a value or a strategy.")

        if self.value is not None and self.strategy is not None:
            raise TransformerError(f"Both a value and a strategy have been selected."
                                   f"Please select either a value or a strategy.")

    # noinspection PyUnresolvedReferences
    def _col_is_categorical_and_is_missing_category(self, col: str, X: pd.DataFrame) -> bool:
        if pd.api.types.is_categorical_dtype(X[col]):
            if self.value_map_[col] not in X[col].cat.categories:
                return True
        return False

    def fit(self, X: pd.DataFrame, y=None):

        self._validate_input()

        if self.strategy:
            impute_values = self.func_map_[self.strategy](X)
            self.value_map_ = {col: impute_values[col] for col in X.columns}

        else:
            self.value_map_ = {col: self.value for col in X.columns}

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            if self._col_is_categorical_and_is_missing_category(col, x_):
                x_[col].cat.add_categories(self.value_map_[col], inplace=True)

        result = x_.fillna(value=self.value_map_)
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
        x_ = X.copy()

        for col in x_.columns:
            x_[col] = pd.Categorical(x_[col],
                                     categories=self.cat_map_[col].categories,
                                     ordered=self.cat_map_[col].ordered)
        return pd.get_dummies(x_)


# noinspection PyUnusedLocal
class FuncTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a given function to each column
    """

    def __init__(self, func: Callable[[DataType], pd.DataFrame], **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = self.func(x_[col], **self.kwargs)
        return x_


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
        x_ = X.reset_index(drop=True)

        x_ts = [t.transform(x_) for _, t in self.transformer_list]
        x_union = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), x_ts)
        return x_union


# noinspection PyUnusedLocal
class Binner(BaseEstimator, TransformerMixin):
    """
    Bins data according to passed bins and labels. Uses pd.cut() under the hood,
    see https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html#pandas-cut
    for further details
    """

    def __init__(self, bins: Union[int, list], labels: list = None):
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = pd.cut(x_[col], bins=self.bins, labels=self.labels)
        return x_


# noinspection PyUnusedLocal
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
        x_ = X.copy()
        for col in x_.columns:
            if self.day:
                x_[f"{col}_day"] = x_[col].dt.day
            if self.month:
                x_[f"{col}_month"] = x_[col].dt.month
            if self.year:
                x_[f"{col}_year"] = x_[col].dt.year
            if self.week:
                x_[f"{col}_week"] = x_[col].dt.week
            x_ = x_.drop(col, axis=1)
        return x_


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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = x_[col].str.upper().map(self.frequencies[col]).fillna(0)
        return x_


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
    the predefined or a callable. If some elements in the row are NaN these
    elements are ignored for the built-in strategies.
    Valid strategies are:
        - sum
        - min
        - max
        - mean
    """

    _func_map = {'sum': np.sum,
                 'min': np.min,
                 'max': np.max,
                 'mean': np.mean}

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
        x_ = x_.apply(self.func, axis=1).to_frame()
        return x_


class Binarize(BaseEstimator, TransformerMixin):
    """
    Binarizer that returns a pandas DataFrame
    """

    def __init__(self, value):
        self.value = value

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        data = np.where(x_ == self.value, 1, 0)
        return pd.DataFrame(data=data, columns=X.columns, index=X.index)
