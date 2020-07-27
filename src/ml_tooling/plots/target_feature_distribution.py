from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from ml_tooling.utils import DataType
from ml_tooling.plots.utils import _plot_barh


class MLToolingError(Exception):
    """Error which occurs when using ML Tooling"""


class VizError(MLToolingError):
    """Error which occurs when using a Visualization"""


def plot_target_feature_distribution(
    target: DataType,
    feature: DataType,
    title: str = "Target feature distribution",
    method: str = "mean",
    ax: plt.Axes = None,
    n_boots: int = 1000,
) -> Axes:
    """
    Creates a plot which compares the mean or median
    of a binary target based on the given category features.

    Parameters
    ----------
    target: np.Array or pd.Series
        Target to compare with feature
    feature: np.Array or pd.Series
        Feature to compare with target
    title: str
        Title of graph
    method: str
        Which method to compare with. Support one of 'median' and 'mean'.
    ax: plt.Axes
        Matplotlib axes to draw the graph on. Creates a new one by default
    n_boots: int
        The number of boostrap iterations to use.
    Returns
    -------
    plt.Axes

    """
    if 0 and 1 not in np.unique(target):
        raise VizError("Target feature distribution plot only works for binary target")

    method_mapping = {"mean": np.mean, "median": np.median}

    selected_method = method_mapping[method]

    feature_categories = np.unique(feature)

    data = np.asarray(
        [
            selected_method(target[feature == category])
            for category in feature_categories
        ]
    )

    final_bootstrap = np.empty([n_boots, len(feature_categories)])
    for i in range(n_boots):
        boots_index = np.random.choice(feature, size=len(feature), replace=True)
        boots_temp = [
            selected_method(target[boots_index][feature[boots_index] == category])
            for category in feature_categories
        ]
        final_bootstrap[i, :] = boots_temp

    percentile = np.percentile(final_bootstrap, (2.5, 97.5), axis=0)

    if ax is None:
        fig, ax = plt.subplots()

    ax = _plot_barh(
        feature_categories,
        data,
        add_label=True,
        title=title,
        x_label=f"Percentage of target compared to {method}",
        y_label="Feature categories",
        ax=ax,
        yerr=percentile,
    )

    ax.axvline(y=selected_method(feature), linestyle="--", color="red")

    return ax
