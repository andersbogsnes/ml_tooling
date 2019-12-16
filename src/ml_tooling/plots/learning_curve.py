from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes, np
from ml_tooling.utils import DataType
from sklearn.model_selection import learning_curve
from sklearn.base import is_classifier


def plot_learning_curve(
    estimator,
    x: pd.DataFrame,
    y: DataType,
    cv: int = 5,
    scoring: str = "default",
    n_jobs: int = -1,
    train_sizes: Sequence = np.linspace(0.1, 1.0, 5),
    ax: Axes = None,
    random_state: int = None,
    title: str = "Learning Curve",
    **kwargs,
) -> Axes:
    """
    Generates a :func:`~sklearn.model_selection.learning_curve` plot,
    used to determine model performance as a function of number of training examples.

    Illustrates whether or not number of training examples is the performance bottleneck.
    Also used to diagnose underfitting or overfitting,
    by seeing how the training set and validation set performance differ.

    Parameters
    ----------
    estimator: sklearn-compatible estimator
        An instance of a sklearn estimator
    x: pd.DataFrame
        DataFrame of features
    y: pd.Series or np.Array
        Target values to predict
    cv: int
        Number of CV iterations to run. Uses a :class:`~sklearn.model_selection.StratifiedKFold` if
        `estimator` is a classifier - otherwise a :class:`~sklearn.model_selection.KFold` is used.
    scoring: str
        Metric to use in scoring - must be a scikit-learn compatible
        :ref:`scoring method<sklearn:scoring_parameter>`
    n_jobs: int
        Number of jobs to use in parallelizing the estimator fitting and scoring
    train_sizes: Sequence of floats
        Percentage intervals of data to use when training
    ax: plt.Axes
        The plot will be drawn on the passed ax - otherwise a new figure and ax will be created.
    random_state: int
        Random state to use in CV splitting
    title: str
        Title to be used on the plot
    kwargs: dict
        Passed along to matplotlib line plots

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    if scoring == "default":
        scoring = "accuracy" if is_classifier(estimator) else "r2"

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        x,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    ax.plot(
        train_sizes, train_scores_mean, label=f"Training {scoring.title()}", **kwargs
    )
    ax.plot(
        train_sizes,
        test_scores_mean,
        label=f"Cross-validated {scoring.title()}",
        **kwargs,
    )
    ax.legend(loc="best")
    ax.set_ylabel(f"{scoring.title()} Score")
    ax.set_xlabel("Number of Examples Used")
    ax.set_title(title)

    return ax
