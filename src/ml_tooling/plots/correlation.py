from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
from ml_tooling.metrics.correlation import target_correlation
from ml_tooling.plots.utils import _plot_barh
from ml_tooling.utils import DataType


def plot_target_correlation(
    features: pd.DataFrame,
    target: DataType,
    method: str = "spearman",
    ax: plt.Axes = None,
    top_n: Union[int, float] = None,
    bottom_n: Union[int, float] = None,
) -> plt.Axes:
    """
    Plot the correlation between each feature
    and the target variable using the given value.

    Also allows selecting how many features to show by
    setting the top_n and/or bottom_n parameters.

    Parameters
    ----------
    features: pd.DataFrame
        Features to plot

    target: np.Array or pd.Series
        Target to calculate correlation with

    method: str
        Which method to use when calculating
        correlation. Supports one of 'pearson', 'spearman', 'kendall'.

    ax: plt.Axes
        Matplotlib axes to draw the graph on. Creates a new one by default

    top_n: int, float
        If top_n is an integer, return top_n features.
        If top_n is a float between (0, 1), return top_n percent features

    bottom_n: int, float
        If bottom_n is an integer, return bottom_n features.
        If bottom_n is a float between (0, 1), return bottom_n percent features


    Returns
    -------
    plt.Axes
    """
    correlation = target_correlation(features, target, method=method, ascending=True)

    if ax is None:
        fig, ax = plt.subplots()

    return _plot_barh(
        correlation.values,
        correlation.index,
        add_label=True,
        title="Feature to Target Correlation",
        x_label=f"{method.title()} Correlation",
        y_label="Feature Labels",
        ax=ax,
        top_n=top_n,
        bottom_n=bottom_n,
    )
