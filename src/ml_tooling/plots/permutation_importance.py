from typing import Union

import pandas as pd
from matplotlib.axes import Axes

from sklearn.inspection import permutation_importance
from ml_tooling.plots.utils import _plot_barh
from ml_tooling.utils import DataType, Estimator
from sklearn.base import is_classifier


def plot_permutation_importance(
    estimator: Estimator,
    x: pd.DataFrame,
    y: DataType,
    scoring="default",
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: int = None,
    ax: Axes = None,
    bottom_n: Union[int, float] = None,
    top_n: Union[int, float] = None,
    add_label: bool = True,
    title: str = "",
    **kwargs,
) -> Axes:
    """
    Calculate permutation importance of dataset x. Permutation importance
    is calculated by randomly permuting a column of input data and scoring the estimator.

    The feature importance is then calculated as the difference between the
    baseline score of all features and the score when data is permuted.

    Note that this approach will inform how important a given input feature is
    - not how important a given category in a feature is for example. This is
    important when doing one-hot encoding, as permutation importance will permute the
    input feature before one-hot encoding.

    Parameters
    ----------
    estimator: Estimator
        Estimator to use to calculate permuted feature importance

    x: DataType
        Features to calculate permuted feature importance for

    y: DataType
        Target to use in scoring

    scoring: str
        Which scoring method to use when assessing changes in score.

    n_repeats: int
        How many times to permute each column.

    n_jobs: int
        How many cores to use in parallel

    random_state: int
        Random state to use in permutations

    ax: Axes
        Matplotlib axes to draw the graph on. Creates a new one by default

    bottom_n: int
        Plot only bottom n features

    top_n: int
        Plot only top n features

    add_label: bool
        Whether or not to plot text labels for the bars

    title: str
        Title to add to the plot

    kwargs: dict
        Any kwargs are passed to matplotlib

    Returns
    -------
    plt.Axes
    """
    if scoring == "default":
        scoring = "accuracy" if is_classifier(estimator) else "r2"

    importances = permutation_importance(
        estimator,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    feature_importance = importances.importances_mean
    labels = x.columns
    plot_title = title if title else f"Feature Importances ({scoring.title()})"
    ax = _plot_barh(
        feature_importance,
        labels,
        add_label=add_label,
        title=plot_title,
        x_label=f"Permuted Feature Importance ({scoring.title()}) Relative to Baseline",
        y_label="Feature Labels",
        ax=ax,
        top_n=top_n,
        bottom_n=bottom_n,
        **kwargs,
    )
    return ax
