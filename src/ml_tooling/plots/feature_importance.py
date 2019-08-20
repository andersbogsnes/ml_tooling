from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..metrics.utils import _is_percent, _sort_values
from .utils import _generate_text_labels
from ..utils import DataType


def plot_feature_importance(
    importance: DataType,
    labels: DataType,
    values: bool = None,
    title: str = None,
    x_label: str = None,
    ax: Axes = None,
    top_n: Union[int, float] = None,
    bottom_n: Union[int, float] = None,
) -> Axes:
    """
    Plot a horizontal bar chart of labelled feature importance

    :param importance:
        Importance measure - typically feature importance or coefficient

    :param labels:
        Name of feature

    :param title:
        Plot title

    :param x_label:
        Plot x-axis label

    :param values:
        Add value labels to end of each bar

    :param ax:
        Pass your own ax

    :param top_n:
        If top_n is an integer, return top_n features
        If top_n is a float between 0 and 1, return top_n percent of features

    :param bottom_n:
        If bottom_n is an integer, return bottom_n features
        If bottom_n is a float between 0 and 1, return bottom_n percent of features


    :return:
        matplotlib.Axes
    """

    if ax is None:
        fig, ax = plt.subplots()

    title = f"Feature Importance" if title is None else title

    if top_n:
        if _is_percent(top_n):
            title = f"{title} - Top {top_n:.0%}"
        else:
            title = f"{title} - Top {top_n}"

    if bottom_n:
        if _is_percent(bottom_n):
            title = f"{title} - Bottom {bottom_n:.0%}"
        else:
            title = f"{title} - Bottom {bottom_n}"

    labels, importance = _sort_values(
        labels, importance, sort="abs", top_n=top_n, bottom_n=bottom_n
    )
    labels, importance = labels[::-1], importance[::-1]
    ax.barh(labels, np.abs(importance))
    ax.set_title(title)
    ax.set_ylabel("Features")
    x_label = "Importance" if x_label is None else x_label
    ax.set_xlabel(x_label)
    if values:
        for i, (x, y) in enumerate(_generate_text_labels(ax, horizontal=True)):
            ax.annotate(
                f"{importance[i]:.2f}",
                (x, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
            )

    return ax
