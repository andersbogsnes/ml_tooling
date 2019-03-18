from functools import reduce

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
