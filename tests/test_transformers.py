import pytest
import pandas as pd
import numpy as np
from ml_tooling.result import CVResult, Result
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from ml_tooling.transformers import (
    Select,
    FillNA,
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
)

from ml_tooling.utils import TransformerError


class TransformerBase:
    @staticmethod
    def create_pipeline(transformer):
        pipe = Pipeline([("transform", transformer), ("clf", DummyClassifier())])
        return pipe

    def create_model(self, base, transformer):
        pipe = self.create_pipeline(transformer)
        model = base(pipe)
        model.config.N_JOBS = 2
        return model

    def create_gridsearch(self, transformer):
        pipe = self.create_pipeline(transformer)
        return GridSearchCV(
            pipe,
            param_grid={"clf__strategy": ["stratified", "most_frequent"]},
            cv=2,
            iid=False,
        )


class TestDFSelector(TransformerBase):
    @pytest.mark.parametrize(
        "container", [["category_a"], "category_a", ("category_a",)]
    )
    def test_df_selector_returns_correct_dataframe(self, categorical, container):
        select = Select(container)
        result = select.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        assert {"category_a"} == set(result.columns)

    def test_df_selector_with_multiple_columns(self, categorical):
        select = Select(["category_a", "category_b"])
        result = select.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        assert {"category_a", "category_b"} == set(result.columns)

    def test_df_selector_raise_missing_column(self, categorical):
        select = Select(["category_a", "category_b", "category_c"])

        with pytest.raises(
            TransformerError, match="The DataFrame does not include the columns:"
        ):
            select.fit_transform(categorical)

    def test_df_selector_works_cross_validated(self, base):
        model = self.create_model(base, Select("sepal length (cm)"))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_df_selector_works_gridsearch(self, base):
        grid = self.create_gridsearch(Select("sepal length (cm)"))
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestFillNA(TransformerBase):
    @pytest.mark.parametrize("value, strategy", [(None, None), (0, "mean")])
    def test_fillna_raises_error(self, numerical_na, value, strategy):
        with pytest.raises(TransformerError):
            FillNA(value=value, strategy=strategy).fit(numerical_na)

    def test_fillna_returns_dataframe_unchanged_if_no_nans(self, categorical):
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
        self, categorical_na, value, strategy, expected
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
        self, numerical_na, value, strategy, expected
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
        self, value, strategy, expected, categorical_na
    ):
        categorical_na["category_a"] = categorical_na["category_a"].astype("category")
        categorical_na["category_b"] = categorical_na["category_b"].astype("category")

        imputer = FillNA(value=value, strategy=strategy)
        result = imputer.fit_transform(categorical_na)

        pd.testing.assert_frame_equal(result, expected, check_categorical=False)

    def test_fillna_works_cross_validated(self, base):
        model = self.create_model(base, FillNA(0))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_fillna_works_gridsearch(self, base):
        grid = self.create_gridsearch(FillNA(0))
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestToCategorical(TransformerBase):
    def test_to_categorical_returns_correct_dataframe(self, categorical):
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

    def test_to_categorical_discards_unseen_values(self, categorical):
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

    def test_to_categorical_works_in_cv(self, base):
        model = self.create_model(base, ToCategorical())
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_to_categorical_works_gridsearch(self, base):
        grid = self.create_gridsearch(ToCategorical())
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestBinner(TransformerBase):
    def test_binner_returns_correctly(self, numerical):
        labels = ["1", "2", "3", "4"]
        binner = Binner(bins=[0, 1, 2, 3, 4], labels=labels)
        result = binner.fit_transform(numerical[["number_a"]])

        assert isinstance(result, pd.DataFrame)
        assert len(numerical) == len(result)
        assert pd.api.types.is_categorical_dtype(result["number_a"])
        assert 4 == len(result["number_a"].cat.categories)
        assert set(labels) == set(result["number_a"].cat.categories)

    def test_binner_returns_nan_on_unseen_data(self, numerical):
        labels = ["1", "2", "3", "4"]
        binner = Binner(bins=[0, 1, 2, 3, 4], labels=labels)
        binner.fit(numerical[["number_a"]])

        new_data = pd.DataFrame({"number_a": [5, 6, 7, 8]})
        result = binner.transform(new_data)

        assert isinstance(result, pd.DataFrame)
        assert len(new_data) == len(result)
        assert len(new_data) == result.isna().sum().sum()

    def test_binner_can_be_used_cv(self, base):
        model = self.create_model(base, Binner(3))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_binner_works_gridsearch(self, base):
        grid = self.create_gridsearch(Binner(3))
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestRenamer(TransformerBase):
    def test_renamer_returns_correctly(self, numerical):
        new_col_names = ["test_a", "test_b"]
        renamer = Renamer(new_col_names)
        result = renamer.fit_transform(numerical)

        assert isinstance(result, pd.DataFrame)
        assert len(numerical) == len(result)
        assert set(new_col_names) == set(result.columns)

    def test_renamer_works_correctly_if_only_given_string(self, numerical):
        single_column = numerical.iloc[:, 1].to_frame()
        renamer = Renamer("test")
        result = renamer.fit_transform(single_column)

        assert isinstance(result, pd.DataFrame)
        assert ["test"] == result.columns
        assert len(numerical) == len(result)

    def test_mismatched_no_of_names_raises(self, numerical):
        new_col_names = ["test_a"]
        renamer = Renamer(new_col_names)
        with pytest.raises(TransformerError):
            renamer.fit_transform(numerical)

    def test_renamer_works_in_cv(self, base):
        model = self.create_model(base, Renamer(["1", "2", "3", "4"]))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_renamer_works_gridsearch(self, base):
        grid = self.create_gridsearch(Renamer(["1", "2", "3", "4"]))
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestDateEncoder(TransformerBase):
    def test_date_encoder_returns_correctly(self, dates):
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

    def test_date_encoder_returns_only_year(self, dates):
        year_coder = DateEncoder(day=False, month=False, week=False, year=True)
        result = year_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" in result.columns
        assert "date_a_day" not in result.columns
        assert "date_a_month" not in result.columns
        assert "date_a_week" not in result.columns

    def test_date_encoder_returns_only_month(self, dates):
        month_coder = DateEncoder(day=False, month=True, week=False, year=False)
        result = month_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" not in result.columns
        assert "date_a_day" not in result.columns
        assert "date_a_month" in result.columns
        assert "date_a_week" not in result.columns

    def test_date_encoder_returns_only_day(self, dates):
        date_coder = DateEncoder(day=True, month=False, week=False, year=False)
        result = date_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" not in result.columns
        assert "date_a_day" in result.columns
        assert "date_a_month" not in result.columns
        assert "date_a_week" not in result.columns

    def test_date_encoder_returns_only_week(self, dates):
        week_coder = DateEncoder(day=False, month=False, week=True, year=False)
        result = week_coder.fit_transform(dates)

        assert isinstance(result, pd.DataFrame)
        assert 1 == len(result.columns)
        assert len(dates) == len(result)
        assert "date_a_year" not in result.columns
        assert "date_a_day" not in result.columns
        assert "date_a_month" not in result.columns
        assert "date_a_week" in result.columns

    def test_date_encoder_works_in_cv(self, dates):
        pipe = self.create_pipeline(DateEncoder())
        score = cross_val_score(pipe, dates, [0, 1, 1], n_jobs=2, cv=2)
        assert 2 == len(score)

    def test_date_encoder_works_in_grid_search(self, dates):
        pipe = self.create_pipeline(DateEncoder())
        grid = GridSearchCV(
            pipe,
            param_grid={"clf__strategy": ["stratified", "most_frequent"]},
            cv=2,
            iid=False,
        )
        grid.fit(dates, [0, 1, 1])
        assert hasattr(grid, "best_score_")


class TestFreqFeature(TransformerBase):
    def test_freqfeature_returns_correctly(self, categorical):
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

    def test_freqfeature_handles_nans_correctly(self, categorical_na):
        freq_feature = FreqFeature()
        result = freq_feature.fit_transform(categorical_na)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical_na) == len(result)
        assert set(categorical_na.columns) == set(result)
        assert 0 == result.isna().sum().sum()
        assert 2 / 3 == result.iloc[0, 0]

    def test_freq_features_returns_0_when_unseen_value_is_given(self, categorical):
        freq_feature = FreqFeature()
        freq_feature.fit(categorical)

        new_data = pd.DataFrame({"category_a": ["a1", "a2", "c25"]})
        result = freq_feature.transform(new_data)

        assert isinstance(result, pd.DataFrame)
        assert len(new_data) == len(result)
        assert 0 == result.iloc[-1, 0]

    def test_freq_feature_can_be_used_in_cross_validation_string_data(
        self, categorical
    ):
        pipe = self.create_pipeline(FreqFeature())
        score = cross_val_score(pipe, categorical, np.array([1, 0, 1, 0]), cv=2)
        assert np.all(score >= 0)

    def test_freq_feature_can_be_used_in_grid_search(self, categorical):
        pipe = self.create_pipeline(FreqFeature())
        model = GridSearchCV(
            pipe, param_grid={"clf__strategy": ["stratified", "most_frequent"]}, cv=2
        )
        model.fit(categorical, [1, 0, 1, 0])
        assert hasattr(model, "best_estimator_")


class TestFeatureUnion(TransformerBase):
    def test_featureunion_returns_concatenated_df(self, categorical, numerical):
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


class TestStandardScaler(TransformerBase):
    def test_standard_scaler_returns_correct_dataframe(self, numerical):
        numerical_scaled = numerical.copy()
        numerical_scaled["number_a"] = (numerical["number_a"] - 2.5) / 1.118033988749895
        numerical_scaled["number_b"] = (numerical["number_b"] - 6.5) / 1.118033988749895

        scaler = DFStandardScaler()
        result = scaler.fit_transform(numerical)

        pd.testing.assert_frame_equal(result, numerical_scaled)

    def test_standard_scaler_works_in_pipeline_with_feature_union(self, numerical):
        numerical_scaled = numerical.copy()
        numerical_scaled["number_a"] = (numerical["number_a"] - 2.5) / 1.118033988749895
        numerical_scaled["number_b"] = (numerical["number_b"] - 6.5) / 1.118033988749895

        union = DFFeatureUnion(
            [("number_a", Select(["number_a"])), ("number_b", Select(["number_b"]))]
        )

        pipeline = make_pipeline(union, DFStandardScaler())
        result = pipeline.fit_transform(numerical)

        pd.testing.assert_frame_equal(result, numerical_scaled)

    def test_standard_scaler_works_in_cv(self, base):
        model = self.create_model(base, DFStandardScaler())
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_standard_scaler_works_in_gridsearch(self, base):
        grid = self.create_gridsearch(DFStandardScaler())
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestDFRowFunc(TransformerBase):
    @pytest.mark.parametrize(
        "strategy, match",
        [
            (None, "No strategy is specified."),
            ("avg", "Strategy avg is not a predefined strategy"),
            (1337, "1337 is not a callable or a string"),
        ],
    )
    def test_dfrowfunc_test_strategy_input(self, strategy, match, numerical_na):
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
        self, numerical_na, strategy, expected
    ):
        dfrowfunc = DFRowFunc(strategy=strategy)
        result = dfrowfunc.fit_transform(numerical_na)
        pd.testing.assert_frame_equal(result, expected)

    def test_dfrowfunc_cross_validates_correctly(self, base):
        model = self.create_model(base, DFRowFunc(strategy="mean"))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_dfrowfunc_works_in_gridsearch(self, base):
        grid = self.create_gridsearch(DFRowFunc(strategy="mean"))
        model = base(grid)
        result = model.score_estimator()
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
    numerical, passed_func, expected, kwargs
):
    transformer = FuncTransformer(passed_func, **kwargs)
    result = transformer.fit_transform(numerical)

    pd.testing.assert_frame_equal(expected, result, check_dtype=False)


class TestFuncTransformer(TransformerBase):
    def test_func_transformer_returns_correctly_on_categorical(self, categorical):
        func_transformer = FuncTransformer(lambda x: x.str.upper())
        result = func_transformer.fit_transform(categorical)

        assert isinstance(result, pd.DataFrame)
        assert len(categorical) == len(result)
        for col in result.columns:
            assert result[col].str.isupper().all()

    def test_func_transformer_can_be_validated(self, base):
        model = self.create_model(base, FuncTransformer(np.sum))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_func_transformer_works_in_gridsearch(self, base):
        grid = self.create_gridsearch(FuncTransformer(np.mean))
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)


class TestBinarize(TransformerBase):
    def test_binarize_returns_correctly_on_categorical_na(self, categorical_na):
        binarize = Binarize(value="a1")
        result = binarize.fit_transform(categorical_na)
        expected = pd.DataFrame(
            {"category_a": [1, 0, 0, 1], "category_b": [0, 0, 0, 0]}
        )

        pd.testing.assert_frame_equal(expected, result, check_dtype=False)

    def test_binarize_returns_correctly_on_numerical_na(self, numerical_na):
        binarize = Binarize(value=2)
        result = binarize.fit_transform(numerical_na)
        expected = pd.DataFrame({"number_a": [0, 1, 0, 0], "number_b": [0, 0, 0, 0]})

        pd.testing.assert_frame_equal(expected, result, check_dtype=False)

    def test_binarize_can_be_used_cv(self, base):
        model = self.create_model(base, Binarize(value="a1"))
        result = model.score_estimator(cv=2)
        assert isinstance(result, CVResult)

    def test_binarize_works_in_gridsearch(self, base):
        grid = self.create_gridsearch(Binarize(value=2))
        model = base(grid)
        result = model.score_estimator()
        assert isinstance(result, Result)
