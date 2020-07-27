import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateEncoder(BaseEstimator, TransformerMixin):
    """
    Converts a date column into multiple day-month-year columns
    """

    def __init__(
        self, day: bool = True, month: bool = True, week: bool = True, year: bool = True
    ):
        """

        Parameters
        ----------
        day: bool
            If True, a new day column will be added.
        month: bool
            If True, a new month column will be added.
        week: bool
            If True, a new week column will be added.
        year: bool
            If True, a new year column will be added.
        """
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
