from typing import Any, Callable, Union

import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.result import Result
from ml_tooling.utils import TransformerError
from ml_tooling.transformers import (
    Select,
    ToCategorical,
    FuncTransformer,
    Binner,
    Renamer,
    DateEncoder,
    FreqFeature,
    DFStandardScaler,
    DFFeatureUnion,
    DFRowFunc,
    Binarize,
    RareFeatureEncoder,
)


def create_pipeline(transformer) -> Pipeline:
    pipe = Pipeline(
        [("transform", transformer), ("clf", DummyClassifier(strategy="prior"))]
    )
    return pipe


def create_model(transformer) -> Model:
    pipe = Pipeline([("transform", transformer)])
    model = Model(DummyClassifier(strategy="prior"), feature_pipeline=pipe)
    model.config.N_JOBS = 2
    return model


def create_gridsearch(transformer) -> GridSearchCV:
    pipe = create_pipeline(transformer)
    return GridSearchCV(
        pipe,
        param_grid={"clf__strategy": ["stratified", "most_frequent"]},
        cv=2,
    )


class TestDFSelector:
    @pytest.mark.parametrize(
        "container", [["category_a"], "category_a", ("category_a",)]
    )
    def test_df_selector_returns_correct_dataframe(
        self, categorical: pd.DataFrame, container
    ):
        select = Select(container)
        result = select.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        assert {"category_a"} == set(result.columns)

    def test_df_selector_with_multiple_columns(self, categorical: pd.DataFrame):
        select = Select(["category_a", "category_b"])
        result = select.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        assert {"category_a", "category_b"} == set(result.columns)

    def test_df_selector_raise_missing_column(self, categorical: pd.DataFrame):
        select = Select(["category_a", "category_b", "category_c"])

        with pytest.raises(
            TransformerError, match="The DataFrame does not include the columns:"
        ):
            select.fit_transform(categorical)

    def test_df_selector_works_cross_validated(self, train_iris_dataset):
        model = create_model(Select("sepal length (cm)"))
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_df_selector_works_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(Select("sepal length (cm)"))
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)

    def test_works_without_args(self):
        assert Select()


class TestToCategorical:
    def test_works_without_args(self):
        assert ToCategorical()

    def test_to_categorical_returns_correct_dataframe(self, categorical: pd.DataFrame):
        to_cat = ToCategorical()
        result = to_cat.fit_transform(categorical)
        expected_cols = [
            "category_a_a1",
            "category_a_a2",
            "category_a_a3",
            "category_b_b1",
            "category_b_b2",
            "category_b_b3",
        ]

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        assert set(expected_cols) == set(result.columns)
        for col in expected_cols:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_to_categorical_discards_unseen_values(self, categorical: pd.DataFrame):
        to_cat = ToCategorical()
        to_cat.fit(categorical)
        new_data = pd.DataFrame(
            {"category_a": ["a1", "a2", "ab1"], "category_b": ["b1", "b2", "ab2"]}
        )

        result = to_cat.transform(new_data)
        expected_cols = [
            "category_a_a1",
            "category_a_a2",
            "category_a_a3",
            "category_b_b1",
            "category_b_b2",
            "category_b_b3",
        ]

        assert isinstance(result, pd.DataFrame)
        assert 0 == result.isna().sum().sum()
        assert set(expected_cols) == set(result.columns)

    def test_to_categorical_works_in_cv(self, train_iris_dataset):
        model = create_model(ToCategorical())
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_to_categorical_works_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(ToCategorical())
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


class TestBinner:
    def test_works_without_args(self):
        assert Binner()

    def test_binner_returns_correctly(self, numerical: pd.DataFrame):
        labels = ["1", "2", "3", "4"]
        binner = Binner(bins=[0, 1, 2, 3, 4], labels=labels)
        result = binner.fit_transform(numerical[["number_a"]])

        assert isinstance(result, pd.DataFrame)
        assert len(numerical) == len(result)
        assert pd.api.types.is_categorical_dtype(result["number_a"])
        assert 4 == len(result["number_a"].cat.categories)
        assert set(labels) == set(result["number_a"].cat.categories)

    def test_binner_returns_nan_on_unseen_data(self, numerical: pd.DataFrame):
        labels = ["1", "2", "3", "4"]
        binner = Binner(bins=[0, 1, 2, 3, 4], labels=labels)
        binner.fit(numerical[["number_a"]])

        new_data = pd.DataFrame({"number_a": [5, 6, 7, 8]})
        result = binner.transform(new_data)

        assert isinstance(result, pd.DataFrame)
        assert len(new_data) == len(result)
        assert len(new_data) == result.isna().sum().sum()

    def test_binner_can_be_used_cv(self, train_iris_dataset):
        model = create_model(Binner(3))
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_binner_works_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(Binner(3))
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


class TestRenamer:
    def test_works_without_args(self):
        assert Renamer()

    def test_renamer_returns_correctly(self, numerical: pd.DataFrame):
        new_col_names = ["test_a", "test_b"]
        renamer = Renamer(new_col_names)
        result = renamer.fit_transform(numerical)

        assert isinstance(result, pd.DataFrame)
        assert len(numerical) == len(result)
        assert set(new_col_names) == set(result.columns)

    def test_renamer_works_correctly_if_only_given_string(
        self, numerical: pd.DataFrame
    ):
        single_column = numerical.iloc[:, 1].to_frame()
        renamer = Renamer("test")
        result = renamer.fit_transform(single_column)

        assert isinstance(result, pd.DataFrame)
        assert ["test"] == result.columns
        assert len(numerical) == len(result)

    def test_mismatched_no_of_names_raises(self, numerical: pd.DataFrame):
        new_col_names = ["test_a"]
        renamer = Renamer(new_col_names)
        with pytest.raises(TransformerError):
            renamer.fit_transform(numerical)

    def test_renamer_works_in_cv(self, train_iris_dataset):
        model = create_model(Renamer(["1", "2", "3", "4"]))
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_renamer_works_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(Renamer(["1", "2", "3", "4"]))
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


class TestDateEncoder:
    def test_works_without_args(self):
        assert DateEncoder()

    def test_date_encoder_returns_correctly(self, dates: pd.DataFrame):
        date_coder = DateEncoder()
        result = date_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 4 == len(result.columns)
        assert len(dates) == len(result)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
        assert "date_a" not in result.columns
        assert "date_a_year" in result.columns
        assert "date_a_day" in result.columns
        assert "date_a_month" in result.columns
        assert "date_a_week" in result.columns

    def test_date_encoder_returns_only_year(self, dates: pd.DataFrame):
        year_coder = DateEncoder(day=False, month=False, week=False, year=True)
        result = year_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" in result.columns
        assert "date_a_day" not in result.columns
        assert "date_a_month" not in result.columns
        assert "date_a_week" not in result.columns

    def test_date_encoder_returns_only_month(self, dates: pd.DataFrame):
        month_coder = DateEncoder(day=False, month=True, week=False, year=False)
        result = month_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" not in result.columns
        assert "date_a_day" not in result.columns
        assert "date_a_month" in result.columns
        assert "date_a_week" not in result.columns

    def test_date_encoder_returns_only_day(self, dates: pd.DataFrame):
        date_coder = DateEncoder(day=True, month=False, week=False, year=False)
        result = date_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" not in result.columns
        assert "date_a_day" in result.columns
        assert "date_a_month" not in result.columns
        assert "date_a_week" not in result.columns

    def test_date_encoder_returns_only_week(self, dates: pd.DataFrame):
        week_coder = DateEncoder(day=False, month=False, week=True, year=False)
        result = week_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" not in result.columns
        assert "date_a_day" not in result.columns
        assert "date_a_month" not in result.columns
        assert "date_a_week" in result.columns

    def test_date_encoder_works_in_cv(self, dates: pd.DataFrame):
        pipe = create_pipeline(DateEncoder())
        score = cross_val_score(pipe, dates, y=[0, 0, 1, 1], n_jobs=2, cv=2)
        assert 2 == len(score)

    def test_date_encoder_works_in_grid_search(self, dates: pd.DataFrame):
        pipe = create_pipeline(DateEncoder())
        grid = GridSearchCV(
            pipe,
            param_grid={"clf__strategy": ["stratified", "most_frequent"]},
            cv=2,
        )
        grid.fit(dates, [0, 0, 1, 1])
        assert hasattr(grid, "best_score_")


class TestFreqFeature:
    def test_works_without_args(self):
        assert FreqFeature()

    def test_freqfeature_returns_correctly(self, categorical: pd.DataFrame):
        freq_feature = FreqFeature()
        result = freq_feature.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        assert set(categorical.columns) == set(result.columns)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
        assert 0.5 == result.iloc[0, 0]
        assert 0.5 == result.iloc[0, 1]
        assert 0.25 == result.iloc[1, 0]
        assert 0.25 == result.iloc[1, 1]
        assert np.all(1.5 == result.sum())

    def test_freqfeature_handles_nans_correctly(self, categorical_na: pd.DataFrame):
        freq_feature = FreqFeature()
        result = freq_feature.fit_transform(categorical_na)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical_na) == len(result)
        assert set(categorical_na.columns) == set(result)
        assert 0 == result.isna().sum().sum()
        assert 2 / 3 == result.iloc[0, 0]

    def test_freq_features_returns_0_when_unseen_value_is_given(
        self, categorical: pd.DataFrame
    ):
        freq_feature = FreqFeature()
        freq_feature.fit(categorical)

        new_data = pd.DataFrame({"category_a": ["a1", "a2", "c25"]})
        result = freq_feature.transform(new_data)

        assert isinstance(result, pd.DataFrame)
        assert len(new_data) == len(result)
        assert 0 == result.iloc[-1, 0]

    def test_freq_feature_can_be_used_in_cross_validation_string_data(
        self, categorical: pd.DataFrame
    ):
        pipe = create_pipeline(FreqFeature())
        score = cross_val_score(pipe, categorical, np.array([1, 0, 1, 0]), cv=2)
        assert np.all(score >= 0)

    def test_freq_feature_can_be_used_in_grid_search(self, categorical: pd.DataFrame):
        pipe = create_pipeline(FreqFeature())
        model = GridSearchCV(
            pipe, param_grid={"clf__strategy": ["stratified", "most_frequent"]}, cv=2
        )
        model.fit(categorical, [1, 0, 1, 0])
        assert hasattr(model, "best_estimator_")


class TestFeatureUnion:
    def test_featureunion_returns_concatenated_df(
        self, categorical: pd.DataFrame, numerical: pd.DataFrame
    ):
        df = pd.concat([categorical, numerical], axis=1)
        first_pipe = make_pipeline(
            Select(["category_a", "category_b"]), ToCategorical()
        )
        union = DFFeatureUnion(
            [("category", first_pipe), ("number", Select(["number_a", "number_b"]))]
        )

        transform_df = union.fit_transform(df)

        assert isinstance(transform_df, pd.DataFrame)
        assert 8 == len(transform_df.columns)
        assert len(df) == len(transform_df)


class TestStandardScaler:
    def test_works_without_args(self):
        assert DFStandardScaler()

    def test_can_reset_scaler_parameters(self):
        scaler = DFStandardScaler()
        scaler.scale_ = 0.7
        scaler.mean_ = 0.5
        scaler._reset()
        assert hasattr(scaler, "scale_") is False

    def test_standard_scaler_with_mean_false_returns_correct_dataframe(
        self, numerical: pd.DataFrame
    ):
        numerical_scaled = numerical.copy()
        numerical_scaled["number_a"] = (numerical["number_a"]) / 1.118033988749895
        numerical_scaled["number_b"] = (numerical["number_b"]) / 1.118033988749895

        scaler = DFStandardScaler(with_mean=False)
        result = scaler.fit_transform(numerical)

        pd.testing.assert_frame_equal(result, numerical_scaled)

    def test_standard_scaler_with_std_false_returns_correct_dataframe(
        self, numerical: pd.DataFrame
    ):
        numerical_scaled = numerical.copy()
        numerical_scaled["number_a"] = numerical["number_a"] - 2.5
        numerical_scaled["number_b"] = numerical["number_b"] - 6.5

        scaler = DFStandardScaler(with_std=False)
        result = scaler.fit_transform(numerical)

        pd.testing.assert_frame_equal(result, numerical_scaled)

    def test_standard_scaler_returns_correct_dataframe(self, numerical: pd.DataFrame):
        numerical_scaled = numerical.copy()
        numerical_scaled["number_a"] = (numerical["number_a"] - 2.5) / 1.118033988749895
        numerical_scaled["number_b"] = (numerical["number_b"] - 6.5) / 1.118033988749895

        scaler = DFStandardScaler()
        result = scaler.fit_transform(numerical)

        pd.testing.assert_frame_equal(result, numerical_scaled)

    def test_standard_scaler_works_in_pipeline_with_feature_union(
        self, numerical: pd.DataFrame
    ):
        numerical_scaled = numerical.copy()
        numerical_scaled["number_a"] = (numerical["number_a"] - 2.5) / 1.118033988749895
        numerical_scaled["number_b"] = (numerical["number_b"] - 6.5) / 1.118033988749895

        union = DFFeatureUnion(
            [("number_a", Select(["number_a"])), ("number_b", Select(["number_b"]))]
        )

        pipeline = make_pipeline(union, DFStandardScaler())
        result = pipeline.fit_transform(numerical)

        pd.testing.assert_frame_equal(result, numerical_scaled)

    def test_standard_scaler_works_in_cv(self, train_iris_dataset):
        model = create_model(DFStandardScaler())
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_standard_scaler_works_in_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(DFStandardScaler())
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


class TestDFRowFunc:
    def test_works_without_args(self):
        assert DFRowFunc()

    @pytest.mark.parametrize(
        "strategy, match",
        [
            (None, "No strategy is specified."),
            ("avg", "Strategy avg is not a predefined strategy"),
            (1337, "1337 is not a callable or a string"),
        ],
    )
    def test_dfrowfunc_test_strategy_input(
        self, strategy: Any, match: str, numerical_na: pd.DataFrame
    ):
        with pytest.raises(TransformerError, match=match):
            DFRowFunc(strategy=strategy).fit(numerical_na)

    @pytest.mark.parametrize(
        "strategy, expected",
        [
            ("sum", pd.DataFrame([5.0, 8.0, 10.0, 4.0])),
            ("min", pd.DataFrame([5.0, 2.0, 3.0, 4.0])),
            ("max", pd.DataFrame([5.0, 6.0, 7.0, 4.0])),
            ("mean", pd.DataFrame([5.0, 4.0, 5.0, 4.0])),
            (lambda x: np.mean(x), pd.DataFrame([5.0, 4.0, 5.0, 4.0])),
            (np.mean, pd.DataFrame([5.0, 4.0, 5.0, 4.0])),
        ],
    )
    def test_dfrowfunc_sum_built_in_and_callable(
        self,
        numerical_na: pd.DataFrame,
        strategy: Union[Callable, str],
        expected: pd.DataFrame,
    ):
        dfrowfunc = DFRowFunc(strategy=strategy)
        result = dfrowfunc.fit_transform(numerical_na)
        pd.testing.assert_frame_equal(result, expected)

    def test_dfrowfunc_cross_validates_correctly(self, train_iris_dataset):
        model = create_model(DFRowFunc(strategy="mean"))
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_dfrowfunc_works_in_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(DFRowFunc(strategy="mean"))
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


def arbitrary_test_func(x, y, z):
    return x - y > z


def wrapper_arbitrary_test_func(df, y, z):
    return df.apply(arbitrary_test_func, y=y, z=z)


@pytest.mark.parametrize(
    "passed_func, expected, kwargs",
    [
        (
            np.mean,
            pd.DataFrame(
                {"number_a": [2.5, 2.5, 2.5, 2.5], "number_b": [6.5, 6.5, 6.5, 6.5]}
            ),
            dict(),
        ),
        (
            lambda x: x * 2,
            pd.DataFrame({"number_a": [2, 4, 6, 8], "number_b": [10, 12, 14, 16]}),
            dict(),
        ),
        (
            wrapper_arbitrary_test_func,
            pd.DataFrame(
                {
                    "number_a": [False, False, False, True],
                    "number_b": [True, True, True, True],
                }
            ),
            {"z": 1, "y": 2},
        ),
    ],
)
def test_func_transformer_returns_correctly_numerical(
    numerical: pd.DataFrame, passed_func: Callable, expected: pd.DataFrame, kwargs: dict
):
    transformer = FuncTransformer(passed_func, **kwargs)
    result = transformer.fit_transform(numerical)

    pd.testing.assert_frame_equal(expected, result, check_dtype=False)


class TestFuncTransformer:
    def test_works_without_args(self):
        assert FuncTransformer()

    def test_func_transformer_returns_correctly_on_categorical(
        self, categorical: pd.DataFrame
    ):
        func_transformer = FuncTransformer(lambda x: x.str.upper())
        result = func_transformer.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        for col in result.columns:
            assert result[col].str.isupper().all()

    def test_func_transformer_can_be_validated(self, train_iris_dataset):
        model = create_model(FuncTransformer(np.sum))
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_func_transformer_works_in_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(FuncTransformer(np.mean))
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


class TestBinarize:
    def test_works_without_args(self):
        assert Binarize()

    def test_binarize_returns_correctly_on_categorical_na(
        self, categorical_na: pd.DataFrame
    ):
        binarize = Binarize(value="a1")
        result = binarize.fit_transform(categorical_na)
        expected = pd.DataFrame(
            {"category_a": [1, 0, 0, 1], "category_b": [0, 0, 0, 0]}
        )

        pd.testing.assert_frame_equal(expected, result, check_dtype=False)

    def test_binarize_returns_correctly_on_numerical_na(
        self, numerical_na: pd.DataFrame
    ):
        binarize = Binarize(value=2)
        result = binarize.fit_transform(numerical_na)
        expected = pd.DataFrame({"number_a": [0, 1, 0, 0], "number_b": [0, 0, 0, 0]})

        pd.testing.assert_frame_equal(expected, result, check_dtype=False)

    def test_binarize_can_be_used_cv(self, train_iris_dataset):
        model = create_model(Binarize(value=1))
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_binarize_works_in_gridsearch(self, train_iris_dataset):
        grid = create_gridsearch(Binarize(value=2))
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)


class TestRareFeatureEncoder:
    @pytest.fixture
    def rare(self) -> RareFeatureEncoder:
        return RareFeatureEncoder(threshold=2, fill_rare="Rare")

    @pytest.fixture
    def categorical_int_and_string(self) -> pd.DataFrame:
        return pd.DataFrame({"categorical": [1, "a", "a", 2, "b", 1]})

    def test_works_without_args(self):
        assert RareFeatureEncoder()

    def test_rare_feature_encoder_that_transformed_data_and_input_data_same_shape(
        self, rare: RareFeatureEncoder, categorical_int_and_string: pd.DataFrame
    ):
        rare.fit(categorical_int_and_string)

        new_data = pd.DataFrame(
            {
                "categorical": [1, 1, 1, "a", "b", "b", 3],
                "numerical": [1, 2, 3, 4, 5, 6, 7],
            }
        )
        result = rare.transform(new_data)
        assert new_data.shape == result.shape

    def test_rare_feature_encoder_returns_correctly_dataframe(
        self, rare: RareFeatureEncoder, categorical_int_and_string: pd.DataFrame
    ):
        rare.fit(categorical_int_and_string)

        new_data = pd.DataFrame(
            {
                "categorical": [2, 2, 2, "a", "b", "b", 3],
                "numerical": [1, 2, 3, 4, 5, 6, 7],
            }
        )
        result = rare.transform(new_data)

        expected = pd.DataFrame(
            {
                "categorical": ["Rare", "Rare", "Rare", "a", "Rare", "Rare", 3],
                "numerical": [1, 2, 3, 4, 5, 6, 7],
            }
        )

        pd.testing.assert_frame_equal(expected, result)

    def test_rare_feature_encoder_doesnt_count_nans(self, rare: RareFeatureEncoder):
        data = pd.DataFrame(
            {
                "categorical": [1, "a", "a", 2, "b", np.nan],
                "numerical": [1, 2, 2, 3, 3, 3],
            }
        )
        rare.fit(data)

        new_data = pd.DataFrame({"categorical": [1, 1, 1, "a", "b", "b", np.nan]})
        result = rare.transform(new_data)

        expected = pd.DataFrame(
            {"categorical": ["Rare", "Rare", "Rare", "a", "Rare", "Rare", np.nan]}
        )

        pd.testing.assert_frame_equal(expected, result)

    def test_rare_feature_encoder_correctly_counts_rare_when_given_percent_threshold(
        self, categorical_int_and_string: pd.DataFrame
    ):
        rare = RareFeatureEncoder(threshold=0.2, fill_rare=99)

        rare.fit(categorical_int_and_string)

        new_data = pd.DataFrame({"categorical": [1, "a", "b", "b", 2]})

        result = rare.transform(new_data)

        expected = pd.DataFrame({"categorical": [1, "a", 99, 99, 99]})

        pd.testing.assert_frame_equal(expected, result)

    def test_rare_feature_encoder_can_be_used_cv(
        self, train_iris_dataset: Dataset, rare: RareFeatureEncoder
    ):
        model = create_model(rare)
        result = model.score_estimator(train_iris_dataset, cv=2)
        assert isinstance(result, Result)

    def test_rare_feature_encoder_works_gridsearch(
        self, train_iris_dataset: Dataset, rare: RareFeatureEncoder
    ):
        grid = create_gridsearch(rare)
        model = Model(grid)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)
