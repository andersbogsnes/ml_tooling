from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from ml_tooling.utils import DataType
from ml_tooling.plots.utils import _plot_barh


def plot_target_feature_distribution(
    target: DataType,
    feature: DataType,
    title: str = "Target feature distribution",
    method: str = "mean",
    ax: plt.Axes = None,
) -> Axes:
    """
    Creates a plot which compares the mean or median
    of a binary target based on the given category features.
    Using np.nanmean or np.nanmedian

    Parameters
    ----------
    target: DataType
        Target to aggregate per feature category
    feature: DataType
        Categorical feature to group by
    title: str
        Title of graph
    method: str
        Which method to compare with. One of 'median' or 'mean'.
    ax: plt.Axes
        Matplotlib axes to draw the graph on. Creates a new one by default
    Returns
    -------
    plt.Axes

    """
    if ax is None:
        fig, ax = plt.subplots()

    agg_func_mapping = {"mean": np.nanmean, "median": np.nanmedian}

    agg_func = agg_func_mapping[method]

    feature_categories = np.sort(np.unique(feature))

    data = np.asarray(
        [agg_func(target[feature == category]) for category in feature_categories]
    )

    ax = _plot_barh(
        values=data,
        label_names=feature_categories,
        add_label=True,
        title=title,
        x_label=f"Target compared to {method}",
        y_label="Feature categories",
        ax=ax,
    )

    ax.axvline(x=agg_func(target), linestyle="--")

    return ax
