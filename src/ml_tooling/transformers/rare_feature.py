import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Replace values which occurs below the chosen threshold.
    The threshold can be a procent or int value.
    The values are replaced with the chosen fill value.
    Column names need to be the same.
    """

    def __init__(self, threshold=0.2, fill_rare="Rare"):
        self.threshold = threshold
        self.fill_rare = fill_rare
        self.mask_dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.columns:
            if self.threshold < 1:
                frequencies = X[col].value_counts(normalize=True)
            else:
                frequencies = X[col].value_counts()

            condition = frequencies < self.threshold
            mask_obs = frequencies[condition].index
            self.mask_dict[col] = dict.fromkeys(mask_obs, self.fill_rare)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x_ = X.copy()

        for col in x_.columns:
            if col in self.mask_dict:
                x_[col] = x_[col].replace(self.mask_dict[col])

        return x_
