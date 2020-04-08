from matplotlib import pyplot as plt

from ml_tooling.plots import plot_residuals, plot_prediction_error
from ml_tooling.plots.viz.baseviz import BaseVisualize


class RegressionVisualize(BaseVisualize):
    """
    Visualization class for Regression models
    """

    def residuals(self, **kwargs) -> plt.Axes:
        """
        Visualizes residuals of a regression estimator.
        Any kwargs are passed onto matplotlib

        Returns
        -------
        matplotlib.Axes
            Plot of the estimator's residuals
        """
        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Residual Plot - {self._estimator_name}"
            y_pred = self._estimator.predict(self._data.test_x)
            return plot_residuals(self._data.test_y, y_pred, title, **kwargs)

    def prediction_error(self, **kwargs) -> plt.Axes:
        """
        Visualizes prediction error of a regression estimator
        Any kwargs are passed onto matplotlib

        Returns
        -------
        matplotlib.Axes
            Plot of the estimator's prediction error
        """

        with plt.style.context(self._config.STYLE_SHEET):
            title = f"Prediction Error - {self._estimator_name}"
            y_pred = self._estimator.predict(self._data.test_x)
            return plot_prediction_error(
                self._data.test_y, y_pred, title=title, **kwargs
            )
