import pathlib

import numpy as np
import pytest
import yaml
from sklearn.linear_model import LinearRegression

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.logging import Log
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

    @pytest.fixture()
    def logs(self, result: Result, tmp_path: pathlib.Path) -> dict:
        log = Log.from_result(result)
        log.save_log(tmp_path)

        with log.output_path.open() as f:
            saved_log = yaml.safe_load(f)

        return saved_log

    def test_cv_result_has_two_cross_val_scores(self, result_cv: Result):
        """Expect a cross-validated result to have two cross_val_scores"""
        assert len(result_cv.metrics.cross_val_scores) == 2

    def test_cv_result_contains_the_same_estimator_as_the_model(
        self, result_cv: Result, model: Model
    ):
        """Expect the Result to have the same estimator as Model"""
        assert result_cv.estimator == model.estimator

    def test_cv_result_has_the_default_r2_metric(self, result_cv: Result):
        """Expect the cross-validated result to use r2 as the default for a regression"""
        assert result_cv.metrics.name == "r2"

    def test_cv_result_should_have_std_deviation_of_cross_validation(
        self, result_cv: Result
    ):
        """Expect cross-validated results to have a std deviation equal to the std deviation
        of the cross val scores
        """
        assert result_cv.metrics.std == np.std(result_cv.metrics.cross_val_scores)

    def test_cv_result_should_have_a_score_equal_to_the_mean_of_cross_val(
        self, result_cv: Result
    ):
        """Expect the cross-validated score to be equal to the mean of cross validated scores"""
        assert result_cv.metrics.score == np.mean(result_cv.metrics.cross_val_scores)

    def test_cv_result_should_have_only_one_metric(self, result_cv: Result):
        """Expect the cross-validated result to have only one metric"""
        assert len(result_cv.metrics) == 1

    def test_cv_results_should_have_a_score(self, result_cv: Result):
        assert result_cv.metrics.score == pytest.approx(0.73037122)

    def test_result_should_not_have_a_cross_val_std(self, result: Result):
        """Expect the result's metrics to not have a cross_val_std attribute"""
        assert result.metrics.std is None

    def test_result_should_have_same_estimator_as_model(
        self, result: Result, model: Model
    ):
        """Expect the result's estimator to be the same as the model"""
        assert result.estimator == model.estimator

    def test_result_score_should_be_correct(self, result: Result):
        """Expect the result's score to be correct"""
        assert result.metrics.score == pytest.approx(0.6844267)

    def test_result_score_should_be_r2_by_default(self, result: Result):
        """Expect the scoring method to be r2 by default"""
        assert result.metrics.name == "r2"

    def test_result_can_return_a_model(self, result: Result):
        """Expect that a result can return a Model"""
        assert result.model.estimator_name == Model(LinearRegression()).estimator_name

    def test_result_logs_metric_score_correctly(self, logs: dict, result: Result):
        """Expect the log to contain the same data as the result"""
        assert logs["metrics"]["r2"] == result.metrics.score

    def test_result_logs_model_name_correctly(self, logs: dict, result: Result):
        """Expect the log to have the correct model name"""
        assert logs["model_name"] == "DemoData_LinearRegression"
