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
    n_boot: int = None,
) -> Axes:
    """
    Creates a plot which compares the mean or median
    of a binary target based on the given category features.

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
    n_boot: int
        The number of bootstrap iterations to use.
    Returns
    -------
    plt.Axes

    """

    if ax is None:
        fig, ax = plt.subplots()

    agg_func_mapping = {"mean": np.nanmean, "median": np.nanmedian}

    agg_func = agg_func_mapping[method]

    feature_categories = np.unique(feature)

    data = np.asarray(
        [agg_func(target[feature == category]) for category in feature_categories]
    )

    if n_boot:

        percentile = np.zeros((2, feature_categories.shape[0]))
        boots_sample = np.random.choice(
            len(target), size=n_boot * target.shape[0], replace=True
        ).reshape((target.shape[0], -1))

        target_boot_sample = np.zeros(
            (boots_sample.shape[0], boots_sample.shape[1]), dtype="int32"
        )
        feature_boot_sample = np.zeros(
            (boots_sample.shape[0], boots_sample.shape[1]), dtype="int32"
        )

        for i in range(len(target)):
            target_boot_sample[i, :] = target[boots_sample[i, :]]
            feature_boot_sample[i, :] = feature[boots_sample[i, :]]

        feature_categories = np.unique(feature_boot_sample)
        for i, category in enumerate(feature_categories):
            data_temp = target_boot_sample[feature_boot_sample == category]
            percentile[:, i] = np.percentile(np.nanmean(data_temp, axis=0), (2.5, 97.5))

    ax = _plot_barh(
        values=data,
        label_names=feature_categories,
        add_label=True,
        title=title,
        x_label=f"Target compared to {method}",
        y_label="Feature categories",
        ax=ax,
        xerr=percentile if n_boot else None,
    )

    ax.axvline(x=agg_func(target), linestyle="--")

    return ax
