from typing import Union, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.axes import Axes
from ml_tooling.metrics.utils import _is_percent, _sort_values
from ml_tooling.utils import DataType
import numpy as np

def _generate_text_labels(ax: plt.Axes, horizontal=False):
    """
    Helper for generating text labels for bar charts

    Parameters
    ----------
    ax: plt.Axes
        Ax which has patches on it

    horizontal: bool
        Whether or not the graph is a horizontal bar chart or a regular bar chart

    Returns
    -------
        x and y values for ax.text
    """
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()

        if horizontal is True:
            x_value = width
            y_value = y + (height / 2)
        else:
            x_value = x + (width / 2)
            y_value = height

        yield x_value, y_value


def _plot_barh(
    values: DataType,
    label_names: DataType,
    add_label: bool = False,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    ax: Optional[Axes] = None,
    top_n: Union[int, float] = None,
    bottom_n: Union[int, float] = None,
    is_percent=False,
    **kwargs,
) -> Axes:
    """
    Utility plot function for plotting a barh that supports plotting only top_n and bottom_n

    Parameters
    ----------
    values: DataType
        Values to plot

    label_names: DataType
        Name of feature

    title: str
        Plot title

    x_label: str
        x-axis label name

    add_label: bool
        Toggle adding value labels to end of each bar

    ax: plt.Axes
        Pass your own ax

    top_n: int or float
        If top_n is an integer, return top_n features
        If top_n is a float between 0 and 1, return top_n percent of features

    bottom_n: int or float
        If bottom_n is an integer, return bottom_n features
        If bottom_n is a float between 0 and 1, return bottom_n percent of features

    is_percent: bool
        Indicates that the x-value is a percentage between 0 and 1. Formats the plot accordingly


    Returns
    -------
        plt.Axes
    """

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if top_n:
        title = (
            f"{title} - Top {top_n:.0%}"
            if _is_percent(top_n)
            else f"{title} - Top {top_n}"
        )

    if bottom_n:
        title = (
            f"{title} - Bottom {bottom_n:.0%}"
            if _is_percent(bottom_n)
            else f"{title} - Bottom {bottom_n}"
        )

    label_names, values = _sort_values(
        label_names, values, abs_sort=True, top_n=top_n, bottom_n=bottom_n
    )

    # Matplotlib barh plots last -> first so we need to reverse the sorted sequence
    label_names, values = label_names[::-1], values[::-1]

    ax.barh(label_names, np.abs(values), **kwargs)
    ax.set_title(title)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if add_label:
        for i, (x, y) in enumerate(_generate_text_labels(ax, horizontal=True)):
            ax.annotate(
                (
                    f"{values.ravel()[i]:.1%}"
                    if is_percent
                    else f"{values.ravel()[i]:.2f}"
                ),
                (x, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
            )

    if is_percent:
        formatter = PercentFormatter(xmax=1)
        ax.xaxis.set_major_formatter(formatter)
    ax.margins(x=0.15)
    return ax
