from typing import Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_tooling.utils import TransformerError


class FillNA(BaseEstimator, TransformerMixin):
    """
    Fills NA values with given value or strategy. Either a value or a strategy
    must be passed.
    """

    def __init__(
        self,
        value: Optional[Union[str, int]] = None,
        strategy: Optional[str] = None,
        indicate_nan: bool = False,
    ):
        """

        Parameters
        ----------
        value: str, int
            A specific value to replace NaNs with.
        strategy: str
            A named strategy to replace NaNs with.
            One of 'mean', 'median', 'most_freq', 'max', 'min'
        indicate_nan: bool
            If True, a new column is added which indicates if a value in a column was missing.

        """

        self.value = value
        self.strategy = strategy
        self.value_map_ = {}
        self.indicate_nan = indicate_nan

        self.func_map_ = {
            "mean": pd.DataFrame.mean,
            "median": pd.DataFrame.median,
            "most_freq": _most_freq,
            "max": pd.DataFrame.max,
            "min": pd.DataFrame.min,
        }

    def _validate_parameters(self):

        if self.value is None and self.strategy is None:
            raise TransformerError(
                "Both value and strategy are set to None."
                "Please select either a value or a strategy."
            )

        if self.value is not None and self.strategy is not None:
            raise TransformerError(
                "Both a value and a strategy have been selected."
                "Please select either a value or a strategy."
            )

    def _col_is_categorical_and_is_missing_category(
        self, col: str, X: pd.DataFrame
    ) -> bool:
        if pd.api.types.is_categorical_dtype(X[col]):
            if self.value_map_[col] not in X[col].cat.categories:
                return True
        return False

    def fit(self, X: pd.DataFrame, y=None):

        self._validate_parameters()

        if self.strategy:
            impute_values = self.func_map_[self.strategy](X)
            try:
                self.value_map_ = {col: impute_values[col] for col in X.columns}
            except KeyError:
                missing_cols = set(X.columns) - set(impute_values.index)
                message = (
                    f"{', '.join(missing_cols)} column/columns "
                    f"have invalid types for strategy = {self.strategy}"
                )
                raise TransformerError(message) from None

        else:
            self.value_map_ = {col: self.value for col in X.columns}

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            if self._col_is_categorical_and_is_missing_category(col, x_):
                x_[col].cat.add_categories(self.value_map_[col], inplace=True)
            if self.indicate_nan:
                is_nan_col_name = f"{col}_is_nan"
                x_[is_nan_col_name] = x_[col].isna().astype(int)

        result = x_.fillna(value=self.value_map_)
        return result


def _most_freq(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mode of X
    :param X:
        DataFrame to calculate mode over
    :return:
        DataFrame of modes
    """
    return pd.DataFrame.mode(X).iloc[0]
