import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.result import Result


class TestPredictionErrorPlot:
    @pytest.fixture(scope="class")
    def regression_result(self) -> Result:
        """Setup a regression result"""
        dataset = load_demo_dataset("boston")
        model = Model(LinearRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, regression_result) -> Axes:
        return regression_result.plot.prediction_error()

    def test_prediction_error_plots_can_be_given_an_ax(self, regression_result: Result):
        """Expect to use the provided ax to plot on"""
        fig, ax = plt.subplots()
        test_ax = regression_result.plot.prediction_error(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_correct_title(self, ax: Axes):
        """Expect plot to have correct title"""
        assert ax.title.get_text() == "Prediction Error - LinearRegression"

    def test_has_correct_ylabel(self, ax: Axes):
        """Expect plot to have correct y label"""
        assert ax.get_ylabel() == "$\\hat{y}$"

    def test_has_correct_xlabel(self, ax: Axes):
        """Expect plot to have correct x label"""
        assert ax.get_xlabel() == "$y$"

    def test_prediction_error_plots_have_correct_data(
        self, ax: Axes, regression_result: Result
    ):
        """Expect plot to have correct data on x and y axes"""
        x = regression_result.plot._data.test_x
        y = regression_result.plot._data.test_y
        y_pred = regression_result.estimator.predict(x)

        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y == ax.collections[0].get_offsets()[:, 0])
        plt.close()
