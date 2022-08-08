import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.result import Result


class TestResidualPlot:
    @pytest.fixture(scope="class")
    def regression_result(self) -> Result:
        """Setup a regression result"""
        dataset = load_demo_dataset("california")
        model = Model(LinearRegression())
        return model.score_estimator(dataset)

    @pytest.fixture(scope="class")
    def ax(self, regression_result) -> Axes:
        """Setup a residuals plot"""
        yield regression_result.plot.residuals()
        plt.close()

    def test_residuals_plots_can_be_given_an_ax(self, regression_result: Result):
        """Expect that the plot will use the axes provided to draw on"""
        fig, ax = plt.subplots()
        test_ax = regression_result.plot.residuals(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_correct_title(self, ax: Axes):
        """Expect the plot to have the correct title"""
        assert "Residual Plot - LinearRegression" == ax.title.get_text()

    def test_has_correct_ylabel(self, ax: Axes):
        """Expect the plot to have correct ylabels"""
        assert ax.get_ylabel() == "Residuals"

    def test_has_correct_xlabel(self, ax: Axes):
        """Expect the plot to have correct xlabels"""
        assert ax.get_xlabel() == "Predicted Value"

    def test_residual_plots_have_correct_data(
        self, ax: Axes, regression_result: Result
    ):
        """Expect the plot to have the correct data"""
        x = regression_result.plot._data.test_x
        y = regression_result.plot._data.test_y
        y_pred = regression_result.estimator.predict(x)
        expected = y_pred - y

        assert np.all(expected == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 0])
        plt.close()
