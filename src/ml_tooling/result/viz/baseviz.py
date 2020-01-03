from typing import Union, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ml_tooling.metrics.permutation_importance import permutation_importance
from ml_tooling.plots import plot_feature_importance, plot_learning_curve, plot_validation_curve
from ml_tooling.utils import _get_estimator_name
from sklearn.base import is_classifier


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, estimator, data, config):
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
            if is_classifier(self._estimator)
            else self._config.REGRESSION_METRIC
        )

    def feature_importance(
        self,
        n_repeats: int = 5,
        values: bool = True,
        top_n: Union[int, float] = None,
        bottom_n: Union[int, float] = None,
        n_jobs: int = None,
        **kwargs,
    ) -> Axes:
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

    def learning_curve(
        self,
        cv: int = 5,
        scoring: str = "default",
        n_jobs: int = None,
        train_sizes: Sequence[float] = np.linspace(0.1, 1.0, 5),
        ax: Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Generates a :func:`~sklearn.model_selection.learning_curve` plot,
        used to determine model performance as a function of number of training examples.

        Illustrates whether or not number of training examples is the performance bottleneck.
        Also used to diagnose underfitting or overfitting,
        by seeing how the training set and validation set performance differ.

        Parameters
        ----------
        cv: int
            Number of CV iterations to run
        scoring: str
            Metric to use in scoring - must be a scikit-learn compatible
            :ref:`scoring method<sklearn:scoring_parameter>`
        n_jobs: int
            Number of jobs to use in parallelizing the estimator fitting and scoring
        train_sizes: Sequence of floats
            Percentage intervals of data to use when training
        ax: plt.Axes
            The plot will be drawn on the passed ax - otherwise a new figure and ax will be created.
        kwargs: dict
            Passed along to matplotlib line plots

        Returns
        -------
        plt.Axes
        """
        
        title = f"Learning Curve - {self._estimator_name}"
        n_jobs = self._config.N_JOBS if n_jobs is None else n_jobs

        with plt.style.context(self._config.STYLE_SHEET):
            ax = plot_learning_curve(
                estimator=self._estimator,
                x=self._data.train_x,
                y=self._data.train_y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                train_sizes=train_sizes,
                title=title,
                ax=ax,
                **kwargs,
            )

        return ax
      
    def validation_curve(
        self,
        param_name: str,
        param_range: Sequence,
        n_jobs: int = None,
        cv: int = 5,
        scoring: str = "default",
        ax: Axes = None,
        **kwargs,
    ) -> Axes:
        """
        Generates a :func:`~sklearn.model_selection.validation_curve` plot,
        graphing the impact of changing a hyperparameter on the scoring metric.

        This lets us examine how a hyperparameter affects
        over/underfitting by examining train/test performance
        with different values of the hyperparameter.

        Parameters
        ----------
        param_name: str
            Name of hyperparameter to plot

        param_range: Sequence
            The individual values to plot for `param_name`

        n_jobs: int
            Number of jobs to use in parallelizing the estimator fitting and scoring

        cv: int
            Number of CV iterations to run. Uses a :class:`~sklearn.model_selection.StratifiedKFold`
            if`estimator` is a classifier - otherwise a :class:`~sklearn.model_selection.KFold`
            is used.

        scoring: str
            Metric to use in scoring - must be a scikit-learn compatible
            :ref:`scoring method<sklearn:scoring_parameter>`

        ax: plt.Axes
            The plot will be drawn on the passed ax - otherwise a new figure and ax will be created.
        
        kwargs: dict
            Passed along to matplotlib line plots
        
        Returns
        -------
        plt.Axes

        """
        n_jobs = self._config.N_JOBS if n_jobs is None else n_jobs
        title = f"Validation Curve - {self._estimator_name}"

        with plt.style.context(self._config.STYLE_SHEET):
            ax = plot_validation_curve(
                self._estimator,
                x=self._data.train_x,
                y=self._data.train_y,
                param_name=param_name,
                param_range=param_range,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                ax=ax,
                title=title,
                **kwargs,
            )
        return ax
