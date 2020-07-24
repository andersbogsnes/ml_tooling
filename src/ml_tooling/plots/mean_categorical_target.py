from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np

from ml_tooling.utils import DataType
from scipy import stats


def plot_mean_categorical_target(
    data: pd.DataFrame,
    col: DataType,
    target: DataType,
    title: str = None,
    width: int = None,
    label: list = None,
) -> Axes:

    title = f"Survival Distributions: {col}" if title is None else title
    width = 0.75 if width is None else width
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title)

    ax1df = data.groupby(by=col)[target]
    std = ax1df.std()
    count = ax1df.count()
    yerr = std / np.sqrt(count) * stats.t.ppf(1 - 0.05 / 2, count - 1)
    ax1df.mean().plot(kind="bar", rot=0, ax=axs[0], yerr=yerr, width=width)
    axs[0].axhline(y=np.mean(data[target]), linestyle="--", color="red")
    axs[0].set_title(f"Percentage {data[target].name} compared to mean")

    data.groupby(target)[col].value_counts().unstack(0).plot(
        kind="bar", rot=0, ax=axs[1], width=width
    )
    axs[1].set_title(f"Distribution of {data[target].name}")
    if label is None:
        axs[1].legend(loc="best")
    else:
        axs[1].legend(label, loc="best")
    return axs
