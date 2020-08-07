import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ToCategorical(BaseEstimator, TransformerMixin):
    """
    Converts a column into a one-hot encoded column through pd.Categorical
    """

    def __init__(self):
        self.cat_map_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_map_ = {col: X[col].astype("category").cat for col in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            x_[col] = pd.Categorical(
                x_[col],
                categories=self.cat_map_[col].categories,
                ordered=self.cat_map_[col].ordered,
            )
        return pd.get_dummies(x_)
