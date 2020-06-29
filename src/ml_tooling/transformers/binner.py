from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Binner(BaseEstimator, TransformerMixin):
    """
    Bins data according to passed bins and labels. Uses :meth:`pandas.cut()` under the hood,
    see for further details
    """

    def __init__(self, bins: Union[int, list] = 5, labels: list = None):
        """

        Parameters
        ----------
        bins: int, list
            The criteria to bin by.
            An int value defines the number of equal-width bins in the range of x. The range of
            x is extended by .1% on each side to include the minimum and maximum values of x.
            If a list is passed, defines the bin edges allowing for non-uniform width and no 
            extension of the range of x is done.
        labels: list
            Specifies the labels for the returned bins. Must be the same length as the
            resulting bins.

        """
        self.bins = bins
        self.labels = labels

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()
        for col in x_.columns:
            x_[col] = pd.cut(x_[col], bins=self.bins, labels=self.labels)
        return x_
