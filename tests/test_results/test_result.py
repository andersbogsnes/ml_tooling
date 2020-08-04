import pathlib

import numpy as np
import pytest
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.logging import Log
from ml_tooling.metrics import Metrics, Metric
from ml_tooling.result import Result


class TestResult:
    @pytest.fixture(scope="class")
    def model(self):
        """Setup a Linear Regression model"""
        return Model(LinearRegression())

    @pytest.fixture(scope="class")
    def result(self, model: Model) -> Result:
        """Setup a Result from a score_estimator without cv"""
        dataset = load_demo_dataset("boston")
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def result_cv(self, model: Model) -> Result:
        """Setup a Result from a cross-validated scoring"""
        dataset = load_demo_dataset("boston")
        return model.score_estimator(dataset, cv=2)

    def test_cv_result_has_two_cross_val_scores(self, result_cv: Result):
        """Expect a cross-validated result to have two cross_val_scores"""
        assert len(result_cv.metrics.cross_val_scores) == 2

    def test_cv_result_contains_the_same_estimator_as_the_model(self,
                                                                result_cv: Result,
                                                                model: Model):
        """Expect the Result to have the same estimator as Model"""
        assert result_cv.estimator == model.estimator

    def test_cv_result_has_the_default_r2_metric(self, result_cv: Result):
        """Expect the cross-validated result to use r2 as the default for a regression"""
        assert result_cv.metrics.name == "r2"

    def test_cv_result_should_have_std_deviation_of_cross_validation(self, result_cv: Result):
        """Expect cross-validated results to have """
        assert result_cv.metrics.cross_val_std == np.std(result_cv.metrics.cross_val_scores)

    def test_cv_result_should_have_a_score_equal_to_the_mean_of_cross_val(self, result_cv: Result):
        assert result_cv.metrics.score == np.mean(result_cv.metrics.cross_val_scores)

    @pytest.mark.parametrize("cv", ["with_cv", "without_cv"])
    def test_linear_model_returns_a_result(
        self, regression: Model, regression_cv: Model, cv: str
    ):
        if cv == "with_cv":
            result = regression_cv.result
            assert isinstance(result, Result)
            assert len(result.metrics[0].cross_val_scores) == 2
            assert result.estimator == regression_cv.estimator
        else:
            result = regression.result
            assert isinstance(result, Result)
            assert hasattr(result, "cross_val_std") is False
            assert regression.estimator == result.estimator

        assert isinstance(result.metrics[0].score, float)
        assert result.metrics[0].name == "r2"
        assert result.estimator.__class__.__name__ == "LinearRegression"

    @pytest.mark.parametrize("cv", ["with_cv", "without_cv"])
    def test_regression_model_returns_a_result(
        self, classifier: Model, classifier_cv: Model, cv: str
    ):
        if cv == "with_cv":
            result = classifier_cv.result
            assert isinstance(result, Result)
            assert 2 == len(result.metrics[0].cross_val_scores)
            assert classifier_cv.estimator == result.estimator

        else:
            result = classifier.result
            assert isinstance(result, Result)
            assert hasattr(result, "cross_val_std") is False
            assert classifier.estimator == result.estimator

        assert result.metrics.score > 0
        assert result.metrics.name == "accuracy"
        assert result.estimator.__class__.__name__ == "LogisticRegression"

    def test_pipeline_regression_returns_correct_result(
        self, pipeline_linear: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_linear)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)
        assert result.estimator == model.estimator
        assert isinstance(result.estimator, Pipeline)

    def test_pipeline_logistic_returns_correct_result(
        self, pipeline_logistic: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        result = model.score_estimator(train_iris_dataset)
        assert isinstance(result, Result)
        assert result.estimator == model.estimator
        assert isinstance(result.estimator, Pipeline)

    def test_result_log_model_returns_correctly(
        self, tmp_path: pathlib.Path, classifier: Model, train_iris_dataset
    ):
        runs = tmp_path / "runs"
        result = Result(
            classifier.estimator,
            data=train_iris_dataset,
            metrics=Metrics([Metric(score=0.7, name="accuracy")]),
        )
        log = Log.from_result(result)
        log.save_log(runs)

        with log.output_path.open(mode="r") as f:
            logged = yaml.safe_load(f)

        assert 0.7 == logged["metrics"]["accuracy"]
        assert logged["model_name"] == "IrisData_LogisticRegression"