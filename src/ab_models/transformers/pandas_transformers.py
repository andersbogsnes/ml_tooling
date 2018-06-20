from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
from functools import reduce


class TransformerError(Exception):
    pass


class Select(BaseEstimator, TransformerMixin):
    """
    Selects columns from DataFrame
    """

    def __init__(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class FillNA(BaseEstimator, TransformerMixin):
    """
    Fills NA values with given value
    """

    def __init__(self, value):
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.fillna(self.value)


class ToCategorical(BaseEstimator, TransformerMixin):
    """
    Converts a column into a one-hot encoded column through pd.Categorical
    """

    def __init__(self):
        self.cat_map_ = None

    def fit(self, X, y=None):
        self.cat_map_ = {col: X[col].astype('category').cat for
                         col in X.columns}
        return self

    def transform(self, X):
        X = X.copy()

        for col in X.columns:
            X[col] = pd.Categorical(X[col],
                                    categories=self.cat_map_[col].categories,
                                    ordered=self.cat_map_[col].ordered)
        return pd.get_dummies(X)


class FuncTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a given function to each column
    """

    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = self.func(X[col])
        return X


class DFFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Merges together two pipelines based on index
    """

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


class Binner(BaseEstimator, TransformerMixin):
    """
    Bins data according to passed bins and labels
    """

    def __init__(self, bins=None, labels=None):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = pd.cut(X[col], bins=self.bins, labels=self.labels)
        return X


class Renamer(BaseEstimator, TransformerMixin):
    """
    Renames columns to passed names
    """

    def __init__(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]

        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if len(self.column_names) != len(X.columns):
            raise TransformerError(f"X has {len(X.columns)} columns - "
                                   f"You provided {len(self.column_names)} column names")
        X.columns = self.column_names
        return X


class DateEncoder(BaseEstimator, TransformerMixin):
    """
    Converts a date column into multiple day-month-year columns
    """

    def __init__(self, day=True, month=True, week=True, year=True):
        self.day = day
        self.month = month
        self.week = week
        self.year = year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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


class FreqFeature(BaseEstimator, TransformerMixin):
    """
    Converts a column into its frequencies
    """

    def __init__(self):
        self.frequencies = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.frequencies[col] = X[col].str.upper().value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].str.upper().map(self.frequencies[col]).fillna(0)
        return X
