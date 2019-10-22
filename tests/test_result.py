import pathlib

import pytest
import yaml
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.logging import Log
from ml_tooling.metrics import Metrics, Metric
from ml_tooling.result import Result, ResultGroup
from ml_tooling.result.viz import create_plotter


class TestResult:
    @pytest.mark.parametrize("cv", ["with_cv", "without_cv"])
    def test_linear_model_returns_a_result(
        self, regression: Model, regression_cv: Model, cv: str
    ):
        if cv == "with_cv":
            result = regression_cv.result
            assert isinstance(result, Result)
            assert len(result.metrics[0].cross_val_scores) == 2
            assert result.model.estimator == regression_cv.estimator
        else:
            result = regression.result
            assert isinstance(result, Result)
            assert hasattr(result, "cross_val_std") is False
            assert regression.estimator == result.model.estimator

        assert isinstance(result.metrics[0].score, float)
        assert result.metrics[0].name == "r2"
        assert result.model.estimator_name == "LinearRegression"

    @pytest.mark.parametrize("cv", ["with_cv", "without_cv"])
    def test_regression_model_returns_a_result(
        self, classifier: Model, classifier_cv: Model, cv: str
    ):
        if cv == "with_cv":
            result = classifier_cv.result
            assert isinstance(result, Result)
            assert 2 == len(result.metrics[0].cross_val_scores)
            assert classifier_cv.estimator == result.model.estimator

        else:
            result = classifier.result
            assert isinstance(result, Result)
            assert hasattr(result, "cross_val_std") is False
            assert classifier.estimator == result.model.estimator

        assert result.metrics.score > 0
        assert result.metrics.name == "accuracy"
        assert result.model.estimator_name == "LogisticRegression"

    def test_pipeline_regression_returns_correct_result(
        self, pipeline_linear: Pipeline, test_dataset: Dataset
    ):
        model = Model(pipeline_linear)
        result = model.score_estimator(test_dataset)
        assert isinstance(result, Result)
        assert result.model.estimator_name == "LinearRegression"
        assert isinstance(result.model.estimator, Pipeline)

    def test_pipeline_logistic_returns_correct_result(
        self, pipeline_logistic: Pipeline, test_dataset: Dataset
    ):
        model = Model(pipeline_logistic)
        result = model.score_estimator(test_dataset)
        assert isinstance(result, Result)
        assert result.model.estimator_name == "LogisticRegression"
        assert isinstance(result.model.estimator, Pipeline)

    def test_result_log_model_returns_correctly(
        self, tmp_path: pathlib.Path, classifier: Model, test_dataset: Dataset
    ):
        runs = tmp_path / "runs"
        result = Result(
            classifier,
            data=test_dataset,
            metrics=Metrics([Metric(score=0.7, name="accuracy")]),
            plot=create_plotter(classifier, test_dataset),
        )
        log = Log.from_result(result)
        run_info = log.save_log(runs)

        with run_info.open(mode="r") as f:
            logged = yaml.safe_load(f)

        assert 0.7 == logged["metrics"]["accuracy"]
        assert logged["model_name"] == "IrisData_LogisticRegression"


class TestResultGroup:
    def test_result_group_proxies_correctly(
        self, test_dataset: Dataset, classifier: Model, classifier_cv: Model
    ):
        result1 = Result.from_model(
            classifier, test_dataset, metrics=Metrics([Metric("accuracy")])
        )
        result2 = Result.from_model(
            classifier_cv, test_dataset, metrics=Metrics([Metric("accuracy")])
        )

        group = ResultGroup([result1, result2])
        result_name = group.model.estimator_name
        assert result_name == "LogisticRegression"

    def test_result_group_implements_indexing_properly(
        self, test_dataset: Dataset, classifier: Model
    ):
        result1 = Result.from_model(
            model=classifier, data=test_dataset, metrics=Metrics.from_list(["accuracy"])
        )

        result2 = Result.from_model(
            model=classifier, data=test_dataset, metrics=Metrics.from_list(["accuracy"])
        )

        group = ResultGroup([result1, result2])
        first = group[0]

        assert first.metrics.score == 0.7368421052631579

    def test_result_group_logs_all_results(
        self, tmp_path: pathlib.Path, test_dataset: Dataset, classifier: Model
    ):
        runs = tmp_path / "runs"
        result1 = Result.from_model(
            model=classifier, data=test_dataset, metrics=Metrics.from_list(["accuracy"])
        )
        result2 = Result.from_model(
            model=classifier, data=test_dataset, metrics=Metrics.from_list(["accuracy"])
        )

        group = ResultGroup([result1, result2])
        group.log_estimator(runs)

        run_files = list(runs.rglob("IrisData_LogisticRegression*"))

        assert len(run_files) == 2
        assert all(("IrisData_LogisticRegression" in file.name for file in run_files))

    def test_logging_a_result_works_as_expected(self, classifier: Model):
        result = classifier.result
        log = result.log()

        assert log.name == "IrisData_LogisticRegression"
        assert "accuracy" in log.metrics
        assert log.estimator_path is None

    def test_resultgroup_sorts_correctly(self, classifier):
        result1 = Result(
            classifier,
            metrics=Metrics.from_dict({"accuracy": 0.9, "roc_auc": 0.1}),
            data=None,
            plot=None,
        )
        result2 = Result(
            classifier,
            metrics=Metrics.from_dict({"accuracy": 0.5, "roc_auc": 0.5}),
            data=None,
            plot=None,
        )

        result3 = Result(
            classifier,
            metrics=Metrics.from_dict({"accuracy": 0.1, "roc_auc": 0.9}),
            data=None,
            plot=None,
        )
        results = ResultGroup([result1, result2, result3])

        results.sort()
        assert results.results == [result1, result2, result3]

        results.sort(by="roc_auc")
        assert results.results == [result3, result2, result1]
