import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DFStandardScaler(BaseEstimator, TransformerMixin):
    """
    Wrapping of the StandardScaler from scikit-learn for Pandas DataFrames. See:
    :class:`~sklearn.preprocessing.StandardScaler`
    """

    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ):
        """
        Parameters
        ----------
        copy: bool
            If True, a copy of the dataframe is made.
        with_mean: bool
            If True, center the data before scaling.
        with_std: bool
            If True, scale the data to unit standard deviation.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.mean_

    def fit(self, X: pd.DataFrame, y=None):
        self._reset()

        if self.with_mean:
            self.mean_ = X.mean()

        if self.with_std:
            self.scale_ = X.std(ddof=0)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy() if self.copy else X
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X
