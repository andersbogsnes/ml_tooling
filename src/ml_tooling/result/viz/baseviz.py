from typing import Union

from matplotlib import pyplot as plt

from ml_tooling.metrics.permutation_importance import permutation_importance
from ml_tooling.plots import plot_feature_importance
from ml_tooling.utils import _get_estimator_name


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, estimator, config, data):
        self._estimator = estimator
        self._estimator_name = _get_estimator_name(estimator)
        self._config = config
        self._data = data

    @property
    def default_metric(self):
        """
        Finds estimator_type for estimator in a BaseVisualize and returns default
        metric for this class stated in .config. If passed estimator is a Pipeline,
        assume last step is the estimator.

        Returns
        -------
        str
            Name of the metric

        """

        return (
            self._config.CLASSIFIER_METRIC
            if self._estimator._estimator_type == "classifier"
            else self._config.REGRESSION_METRIC
        )

    def feature_importance(
        self,
        n_repeats: int = 5,
        values: bool = True,
        top_n: Union[int, float] = None,
        bottom_n: Union[int, float] = None,
        n_jobs=None,
        **kwargs,
    ) -> plt.Axes:
        """
        Visualizes feature importance of the estimator through permutation.

        Parameters
        ----------
        n_repeats : int
            Number of times to permute a feature

        values : bool
            Toggles value labels on end of each bar

        top_n: int, float
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        bottom_n: int, float
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        n_jobs: int
            Overwrites N_JOBS from settings. Useful if data is to big to fit
            in memory multiple times.

        kwargs: dict
            Passed to matplotlib

        Returns
        -------
            matplotlib.Axes
        """

        n_jobs = self._config.N_JOBS if n_jobs is None else n_jobs
        title = f"Feature Importance - {self._estimator_name}"
        result = permutation_importance(
            estimator=self._estimator,
            X=self._data.x,
            y=self._data.y,
            scoring=self.default_metric,
            n_repeats=n_repeats,
            random_state=self._config.RANDOM_STATE,
            n_jobs=n_jobs,
        )
        labels = self._data.train_x.columns

        with plt.style.context(self._config.STYLE_SHEET):
            return plot_feature_importance(
                result.importances_mean,
                labels,
                values=values,
                title=title,
                x_label="Permuted Feature Importance Relative to Baseline",
                top_n=top_n,
                bottom_n=bottom_n,
                **kwargs,
            )
