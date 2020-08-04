import numpy as np
from matplotlib import pyplot as plt

from ml_tooling import Model


class TestResidualPlot:
    def test_residuals_plots_can_be_given_an_ax(self, regression: Model):
        fig, ax = plt.subplots()
        test_ax = regression.result.plot.residuals(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_residual_plots_have_correct_data(self, regression: Model):
        ax = regression.result.plot.residuals()
        x, y = regression.result.plot._data.test_x, regression.result.plot._data.test_y
        y_pred = regression.result.estimator.predict(x)
        expected = y_pred - y

        assert "Residual Plot - LinearRegression" == ax.title._text
        assert "Residuals" == ax.get_ylabel()
        assert "Predicted Value" == ax.get_xlabel()

        assert np.all(expected == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 0])
        plt.close()