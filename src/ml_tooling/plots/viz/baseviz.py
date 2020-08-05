from typing import Union, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.base import is_classifier

from ml_tooling.config import MPL_STYLESHEET, config
from ml_tooling.plots import (
    plot_feature_importance,
    plot_learning_curve,
    plot_validation_curve,
)
from ml_tooling.plots.permutation_importance import plot_permutation_importance
from ml_tooling.utils import _get_estimator_name


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, estimator, data):
        self._estimator = estimator
        self._estimator_name = _get_estimator_name(estimator)
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
            config.CLASSIFIER_METRIC
            if is_classifier(self._estimator)
            else config.REGRESSION_METRIC
        )

    def feature_importance(
        self,
        top_n: Union[int, float] = None,
        bottom_n: Union[int, float] = None,
        class_name: int = None,
        add_label: bool = True,
        ax: Axes = None,
        **kwargs,
    ) -> Axes:
        """
        Visualizes feature importance of the estimator through permutation.

        Parameters
        ----------

        top_n: int, float
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        bottom_n: int, float
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        class_name: int, optional
            In a multi-class setting, plot the feature importances for the given label

        add_label : bool
            Toggles value labels on end of each bar

        ax: Axes
            Draws graph on passed ax - otherwise creates new ax

        kwargs: dict
            Passed to plt.barh

        Returns
        -------
            matplotlib.Axes
        """

        title = f"Feature Importances - {self._estimator_name}"
        title = f"{title} - Class {class_name}" if class_name else title

        with plt.style.context(MPL_STYLESHEET):
            return plot_feature_importance(
                estimator=self._estimator,
                x=self._data.train_x,
                ax=ax,
                class_name=class_name,
                bottom_n=bottom_n,
                top_n=top_n,
                add_label=add_label,
                title=title,
                **kwargs,
            )

    def permutation_importance(
        self,
        n_repeats: int = 5,
        scoring: str = "default",
        top_n: Union[int, float] = None,
        bottom_n: Union[int, float] = None,
        add_label: bool = True,
        n_jobs: int = None,
        ax: Axes = None,
        **kwargs,
    ) -> Axes:
        """
        Visualizes feature importance of the estimator through permutation.

        Parameters
        ----------
        n_repeats : int
            Number of times to permute a feature

        scoring: str
            Metric to use in scoring - must be a scikit-learn compatible
            :ref:`scoring method <sklearn:scoring_parameter>`

        top_n: int, float
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        bottom_n: int, float
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        add_label : bool
            Toggles value labels on end of each bar

        ax: Axes
            Draws graph on passed ax - otherwise creates new ax

        n_jobs: int, optional
            Number of parallel jobs to run. Defaults to N_JOBS setting in config.

        kwargs: dict
            Passed to plt.barh

        Returns
        -------
            matplotlib.Axes
        """
        n_jobs = config.N_JOBS if n_jobs is None else n_jobs
        scoring = self.default_metric if scoring == "default" else scoring
        title = f"Permutation Importances ({scoring.title()}) - {self._estimator_name}"

        with plt.style.context(MPL_STYLESHEET):
            return plot_permutation_importance(
                estimator=self._estimator,
                x=self._data.train_x,
                y=self._data.train_y,
                scoring=scoring,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                random_state=config.RANDOM_STATE,
                ax=ax,
                bottom_n=bottom_n,
                top_n=top_n,
                add_label=add_label,
                title=title,
                **kwargs,
            )

    def learning_curve(
        self,
        cv: int = None,
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
        n_jobs = config.N_JOBS if n_jobs is None else n_jobs
        cv = config.CROSS_VALIDATION if cv is None else cv

        with plt.style.context(MPL_STYLESHEET):
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
        cv: int = None,
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
            Number of CV iterations to run. Defaults to value in `Model.config`.
            Uses a :class:`~sklearn.model_selection.StratifiedKFold`
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
        n_jobs = config.N_JOBS if n_jobs is None else n_jobs
        cv = config.CROSS_VALIDATION if cv is None else cv
        title = f"Validation Curve - {self._estimator_name}"

        with plt.style.context(MPL_STYLESHEET):
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
