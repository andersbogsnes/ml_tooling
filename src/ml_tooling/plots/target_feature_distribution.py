from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np

from ml_tooling.utils import DataType
from scipy import stats


def target_feature_distribution(
    data: pd.DataFrame,
    targets: DataType,
    features: DataType,
    title: str = None,
    labels: list = None,
    median: bool = False,
) -> Axes:

    """
    Creates two plots, one which compares the mean or median
    of a target based on the given features and another which shows
    the distribution of the target based on the given features

    :param data:
        DataFrame containing the data

    :param targets:
        The name of the column containing the targets

    :param features:
        The name of the column containing the features

    :param title:
        Title for plot

    :param labels:
        Pass custom list of labels

    :param median:
        Whether to compare using median or mean, true for median

    :return:
        matplotlib.Axes
    """

    title = f"Survival Distributions: {targets}" if title is None else title
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    ax1df = data.groupby(by=targets)[features]
    std = ax1df.std()
    count = ax1df.count()
    yerr = std / np.sqrt(count) * stats.t.ppf(1 - 0.05 / 2, count - 1)
    if median:
        ax1df.median().plot(kind="bar", rot=0, ax=axs[0], yerr=yerr)
        axs[0].axhline(y=np.median(data[features]), linestyle="--", color="red")
        axs[0].set_title(f"Percentage {data[features].name} compared to median")
    else:
        ax1df.mean().plot(kind="bar", rot=0, ax=axs[0], yerr=yerr)
        axs[0].axhline(y=np.mean(data[features]), linestyle="--", color="red")
        axs[0].set_title(f"Percentage {data[features].name} compared to mean")

    # Is this plot even needed? It doesn't add anything related to the issue
    data.groupby(features)[targets].value_counts().unstack(0).plot(
        kind="bar", rot=0, ax=axs[1]
    )
    axs[1].set_title(f"Distribution of {data[features].name}")
    if labels is None:
        axs[1].legend(loc="best")
    else:
        axs[1].legend(labels, loc="best")
    return axs
