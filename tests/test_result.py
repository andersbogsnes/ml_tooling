import pathlib

import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.logging.log_estimator import save_log
from ml_tooling.result import CVResult, Result, ResultGroup


class TestResult:
    @pytest.mark.parametrize("cv", ["with_cv", "without_cv"])
    def test_linear_model_returns_a_result(self, regression, regression_cv, cv):
        if cv == "with_cv":
            result = regression_cv.result
            assert isinstance(result, CVResult)
            assert len(result.cross_val_scores) == 2
            assert result.cv == 2
            assert "2-fold Cross-validated" in result.__repr__()
            assert result.model.estimator == regression_cv.estimator
        else:
            result = regression.result
            assert isinstance(result, Result)
            assert hasattr(result, "cross_val_std") is False
            assert regression.estimator == result.model.estimator

        assert isinstance(result.score, float)
        assert result.metric == "r2"
        assert result.model.estimator_name == "LinearRegression"

    @pytest.mark.parametrize("cv", ["with_cv", "without_cv"])
    def test_regression_model_returns_a_result(self, classifier, classifier_cv, cv):
        if cv == "with_cv":
            result = classifier_cv.result
            assert isinstance(result, CVResult)
            assert 2 == len(result.cross_val_scores)
            assert 2 == result.cv
            assert "2-fold Cross-validated" in result.__repr__()
            assert classifier_cv.estimator == result.model.estimator

        else:
            result = classifier.result
            assert isinstance(result, Result)
            assert hasattr(result, "cross_val_std") is False
            assert classifier.estimator == result.model.estimator

        assert result.score > 0
        assert result.metric == "accuracy"
        assert result.model.estimator_name == "LogisticRegression"

    def test_pipeline_regression_returns_correct_result(
        self, base, pipeline_linear, test_dataset
    ):
        model = base(pipeline_linear)
        result = model.score_estimator(test_dataset)
        assert isinstance(result, Result)
        assert result.model.estimator_name == "LinearRegression"
        assert isinstance(result.model.estimator, Pipeline)

    def test_pipeline_logistic_returns_correct_result(
        self, base, pipeline_logistic, test_dataset
    ):
        model = base(pipeline_logistic)
        result = model.score_estimator(test_dataset)
        assert isinstance(result, Result)
        assert result.model.estimator_name == "LogisticRegression"
        assert isinstance(result.model.estimator, Pipeline)

    def test_cvresult_equality_operators(self, classifier, test_dataset):
        first_result = CVResult(
            model=classifier, data=test_dataset, cross_val_scores=[2, 2]
        )
        second_result = CVResult(
            model=classifier, data=test_dataset, cross_val_scores=[1, 1]
        )

        assert first_result > second_result

    def test_result_equality_operators(self, classifier, test_dataset):
        first_result = Result(classifier, data=test_dataset, score=0.7)
        second_result = Result(classifier, data=test_dataset, score=0.5)

        assert first_result > second_result

    def test_max_works_with_cv_result(self, classifier, test_dataset):
        first_result = CVResult(classifier, data=test_dataset, cross_val_scores=[2, 2])
        second_result = CVResult(classifier, data=test_dataset, cross_val_scores=[1, 1])

        max_result = max([first_result, second_result])

        assert first_result is max_result

    def test_max_works_with_result(self, classifier, test_dataset):
        first_result = Result(classifier, data=test_dataset, score=0.7)
        second_result = Result(classifier, data=test_dataset, score=0.5)

        max_result = max([first_result, second_result])

        assert first_result is max_result

    def test_result_log_model_returns_correctly(self, tmpdir, classifier, test_dataset):
        runs = tmpdir.mkdir("runs")
        result = Result(classifier, data=test_dataset, score=0.7, metric="accuracy")
        log = result.dump(runs)
        run_info = save_log(log, runs)

        with run_info.open(mode="r") as f:
            logged = yaml.safe_load(f)

        assert 0.7 == logged["metrics"]["accuracy"]
        assert logged["model_name"] == "IrisData_LogisticRegression"

    @pytest.mark.parametrize("with_cv", [True, False])
    def test_result_to_dataframe_returns_correct(
        self, base, pipeline_forest_classifier, with_cv, test_dataset
    ):
        model = base(pipeline_forest_classifier)

        if with_cv:
            result = model.score_estimator(test_dataset, cv=2)

        else:
            result = model.score_estimator(test_dataset)

        df = result.to_dataframe()
        assert 1 == len(df)
        assert "score" in df.columns
        assert "clf__max_depth" in df.columns

    def test_cv_result_with_cross_val_score_returns_correct(
        self, base, pipeline_forest_classifier, test_dataset
    ):
        model = base(pipeline_forest_classifier)
        result = model.score_estimator(test_dataset, cv=2)
        df = result.to_dataframe(cross_val_score=True)
        assert 2 == len(df)
        assert df.loc[0, "score"] != df.loc[1, "score"]
        assert "cv" in df.columns
        assert "cross_val_std" in df.columns


class TestResultGroup:
    def test_result_group_proxies_correctly(self, test_dataset: Dataset):
        result1 = Result(Model(RandomForestClassifier()), test_dataset, 2)
        result2 = Result(Model(LogisticRegression()), test_dataset, 1)

        group = ResultGroup([result1, result2])
        result_name = group.model.estimator_name
        assert "RandomForestClassifier" == result_name

    def test_result_group_sorts_before_proxying(self, test_dataset: Dataset):
        result1 = Result(Model(RandomForestClassifier()), test_dataset, 2)
        result2 = Result(Model(LogisticRegression()), test_dataset, 1)

        group = ResultGroup([result2, result1])
        result_name = group.model.estimator_name

        assert "RandomForestClassifier" == result_name

    def test_result_group_to_frame_has_correct_num_rows(self, test_dataset: Dataset):
        result1 = Result(Model(RandomForestClassifier()), test_dataset, 2)
        result2 = Result(Model(RandomForestClassifier()), test_dataset, 1)

        group = ResultGroup([result2, result1])
        df = group.to_dataframe()

        assert 2 == len(df)
        assert 19 == len(df.columns)

        df_no_params = group.to_dataframe(params=False)

        assert 2 == len(df_no_params)
        assert 2 == len(df_no_params.columns)

    def test_result_cv_group_to_frame_has_correct_num_rows(self, test_dataset: Dataset):
        result1 = CVResult(
            Model(RandomForestClassifier()),
            data=test_dataset,
            cv=2,
            cross_val_scores=[0.5, 0.5],
        )
        result2 = CVResult(
            Model(RandomForestClassifier()),
            data=test_dataset,
            cv=2,
            cross_val_scores=[0.6, 0.6],
        )

        group = ResultGroup([result1, result2])
        df = group.to_dataframe()

        assert 2 == len(df)
        assert 21 == len(df.columns)

        df_no_params = group.to_dataframe(params=False)

        assert 2 == len(df_no_params)
        assert 4 == len(df_no_params.columns)

    def test_result_cv_group_implements_len_properly(self, test_dataset: Dataset):
        result1 = CVResult(
            Model(RandomForestClassifier()),
            test_dataset,
            cv=2,
            cross_val_scores=[0.5, 0.5],
        )
        result2 = CVResult(
            Model(RandomForestClassifier()),
            test_dataset,
            cv=2,
            cross_val_scores=[0.6, 0.6],
        )

        group = ResultGroup([result1, result2])
        assert 2 == len(group)

    def test_result_group_implements_mean_correctly(self, test_dataset: Dataset):
        result1 = Result(Model(RandomForestClassifier()), test_dataset, 2)
        result2 = Result(Model(RandomForestClassifier()), test_dataset, 1)

        group = ResultGroup([result1, result2])
        assert 1.5 == group.mean_score()

    def test_result_group_implements_indexing_properly(self, test_dataset: Dataset):
        result1 = Result(Model(RandomForestClassifier()), test_dataset, 2)
        result2 = Result(Model(RandomForestClassifier()), test_dataset, 1)

        group = ResultGroup([result1, result2])
        first = group[0]

        assert 2 == first.score

    def test_result_group_dir_call_includes_correct_methods(
        self, test_dataset: Dataset
    ):
        result1 = Result(Model(RandomForestClassifier()), test_dataset, 2)
        result2 = Result(Model(RandomForestClassifier()), test_dataset, 1)

        group = ResultGroup([result1, result2])
        options_list = dir(group)

        assert "to_dataframe" in options_list
        assert "plot" in options_list

    def test_result_group_logs_all_results(
        self, tmp_path: pathlib.Path, test_dataset: Dataset
    ):
        runs = tmp_path / "runs"
        result1 = Result(
            Model(RandomForestClassifier()), test_dataset, 2, metric="accuracy"
        )
        result2 = Result(
            Model(RandomForestClassifier()), test_dataset, 1, metric="accuracy"
        )

        group = ResultGroup([result1, result2])
        group.log_estimator(runs)

        run_files = list(runs.rglob("IrisData_RandomForestClassifier*"))

        assert len(run_files) == 2
        assert all(
            ("IrisData_RandomForestClassifier" in file.name for file in run_files)
        )
