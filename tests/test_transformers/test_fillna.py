from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.result import Result, ResultGroup
from ml_tooling.transformers import FillNA
from ml_tooling.utils import TransformerError


class TestFillNA:
    @pytest.fixture()
    def pipeline(self):
        return Pipeline([("fillna", FillNA(0))])

    @pytest.fixture()
    def model(self, pipeline):
        return Model(DummyClassifier(strategy="stratified"), feature_pipeline=pipeline)

    @pytest.mark.parametrize("value, strategy", [(None, None), (0, "mean")])
    def test_fillna_raises_error(self, numerical_na: pd.DataFrame, value, strategy):
        with pytest.raises(TransformerError):
            FillNA(value=value, strategy=strategy).fit(numerical_na)

    def test_fillna_returns_dataframe_unchanged_if_no_nans(
        self, categorical: pd.DataFrame
    ):
        imputer = FillNA("Unknown")
        result = imputer.fit_transform(categorical)
        pd.testing.assert_frame_equal(result, categorical)

    @pytest.mark.parametrize(
        "value, strategy, expected",
        [
            (
                "Unknown",
                None,
                pd.DataFrame(
                    {
                        "category_a": ["a1", "Unknown", "a3", "a1"],
                        "category_b": ["Unknown", "b2", "b3", "b1"],
                    }
                ),
            ),
            (
                None,
                "most_freq",
                pd.DataFrame(
                    {
                        "category_a": ["a1", "a1", "a3", "a1"],
                        "category_b": ["b1", "b2", "b3", "b1"],
                    }
                ),
            ),
        ],
    )
    def test_fillna_imputes_categorical_na_correct(
        self, categorical_na: pd.DataFrame, value: Any, strategy: Any, expected: Any
    ):
        imputer = FillNA(value=value, strategy=strategy)
        result = imputer.fit_transform(categorical_na)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "value, strategy, expected",
        [
            (
                None,
                "mean",
                pd.DataFrame(
                    {"number_a": [3.0, 2.0, 3.0, 4.0], "number_b": [5.0, 6.0, 7.0, 6.0]}
                ),
            ),
            (
                None,
                "median",
                pd.DataFrame(
                    {"number_a": [3.0, 2.0, 3.0, 4.0], "number_b": [5.0, 6.0, 7.0, 6.0]}
                ),
            ),
            (
                None,
                "max",
                pd.DataFrame(
                    {"number_a": [4.0, 2.0, 3.0, 4.0], "number_b": [5.0, 6.0, 7.0, 7.0]}
                ),
            ),
            (
                None,
                "min",
                pd.DataFrame(
                    {"number_a": [2.0, 2.0, 3.0, 4.0], "number_b": [5.0, 6.0, 7.0, 5.0]}
                ),
            ),
        ],
    )
    def test_fill_na_imputes_numerical_na_correct(
        self,
        numerical_na: pd.DataFrame,
        value: None,
        strategy: str,
        expected: pd.DataFrame,
    ):
        imputer = FillNA(value=value, strategy=strategy)
        result = imputer.fit_transform(numerical_na)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "value, strategy, expected",
        [
            (
                "Unknown",
                None,
                pd.DataFrame(
                    {
                        "category_a": ["a1", "Unknown", "a3", "a1"],
                        "category_b": ["Unknown", "b2", "b3", "b1"],
                    },
                    dtype="category",
                ),
            ),
            (
                None,
                "most_freq",
                pd.DataFrame(
                    {
                        "category_a": ["a1", "a1", "a3", "a1"],
                        "category_b": ["b1", "b2", "b3", "b1"],
                    },
                    dtype="category",
                ),
            ),
        ],
    )
    def test_fillna_imputes_pandas_categorical_correct(
        self,
        value: Any,
        strategy: Any,
        expected: pd.DataFrame,
        categorical_na: pd.DataFrame,
    ):
        categorical_na["category_a"] = categorical_na["category_a"].astype("category")
        categorical_na["category_b"] = categorical_na["category_b"].astype("category")

        imputer = FillNA(value=value, strategy=strategy)
        result = imputer.fit_transform(categorical_na)

        pd.testing.assert_frame_equal(result, expected, check_categorical=False)

    def test_fillna_works_cross_validated(self, train_iris_dataset, model):
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_fillna_works_gridsearch(self, train_iris_dataset, model):
        best_estimator, results = model.gridsearch(
            train_iris_dataset,
            param_grid={"estimator__strategy": ["stratified", "prior"]},
        )
        assert isinstance(results, ResultGroup)

    def test_fillna_raises_when_imputing_numerically_on_strings(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "status": ["OK", "Error", "OK", "Error"],
                "sales": [2000, 3000, 4000, np.nan],
            }
        )
        fill_na = FillNA(strategy="mean")
        with pytest.raises(
            TransformerError,
            match="column/columns have invalid types for strategy = mean",
        ):
            fill_na.fit_transform(df)

        df["new_col"] = ["One", "Two", "Three", "Four"]

        with pytest.raises(
            TransformerError,
            match="column/columns have invalid types for strategy = mean",
        ):
            fill_na.fit_transform(df)

    def test_fillna_adds_is_na_column_when_imputing(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "sales": [2000, 3000, 4000, np.nan]})
        fill_na = FillNA(strategy="mean", indicate_nan=True)
        expected_cols = ["id", "sales", "id_is_nan", "sales_is_nan"]
        result = fill_na.fit_transform(df)
        assert result.columns.tolist() == expected_cols
        assert np, all(result["sales_isna"] == [0, 0, 0, 1])
