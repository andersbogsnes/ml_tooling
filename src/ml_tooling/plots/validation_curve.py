from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.base import is_classifier
from sklearn.model_selection import validation_curve

from ml_tooling.utils import DataType


def plot_validation_curve(
    estimator,
    x: pd.DataFrame,
    y: DataType,
    param_name: str,
    param_range: Sequence,
    cv: int = 5,
    scoring: str = "default",
    n_jobs: int = -1,
    ax: Axes = None,
    title: str = "",
    **kwargs,
) -> Axes:
    """
    Plots a :func:`~sklearn.model_selection.validation_curve`, graphing the impact
    of changing a hyperparameter on the scoring metric.

    This lets us examine how a hyperparameter affects
    over/underfitting by examining train/test performance
    with different values of the hyperparameter.


    Parameters
    ----------
    estimator: sklearn-compatible estimator
        An instance of a sklearn estimator

    x: pd.DataFrame
        DataFrame of features

    y: pd.Series or np.Array
        Target values to predict

    param_name: str
        Name of hyperparameter to plot

    param_range: Sequence
        The individual values to plot for `param_name`

    cv: int
        Number of CV iterations to run. Uses a :class:`~sklearn.model_selection.StratifiedKFold` if
        `estimator` is a classifier - otherwise a :class:`~sklearn.model_selection.KFold` is used.

    scoring: str
        Metric to use in scoring - must be a scikit-learn compatible
        :ref:`scoring method<sklearn:scoring_parameter>`

    n_jobs: int
        Number of jobs to use in parallelizing the estimator fitting and scoring

    ax: plt.Axes
        The plot will be drawn on the passed ax - otherwise a new figure and ax will be created.

    title: str
        Title to be used on the plot

    kwargs: dict
        Passed along to matplotlib line plots

    Returns
    -------
    plt.Axes
    """
    if scoring == "default":
        scoring = "accuracy" if is_classifier(estimator) else "r2"

    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=x,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if ax is None:
        fig, ax = plt.subplots()

    title = "Validation Curve" if title is None else title

    ax.plot(
        param_range, train_scores_mean, label=f"Training {scoring.title()}", **kwargs
    )
    ax.fill_between(
        param_range,
        train_scores_mean + train_scores_std,
        train_scores_mean - train_scores_std,
        alpha=0.2,
    )

    ax.plot(param_range, test_scores_mean, label=f"Test {scoring.title()}", **kwargs)
    ax.fill_between(
        param_range,
        test_scores_mean + test_scores_std,
        test_scores_mean - test_scores_std,
        alpha=0.2,
    )

    ax.set_title(title)
    ax.set_ylabel(f"{scoring.title()} Score")
    ax.set_xlabel(f"{param_name.replace('_', ' ').title()}")
    ax.legend(loc="best")
    return ax
