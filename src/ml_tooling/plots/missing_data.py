from typing import Optional, Union

import pandas as pd
from ml_tooling.plots.utils import _plot_barh
import matplotlib.pyplot as plt


def plot_missing_data(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    top_n: Optional[Union[int, float]] = None,
    bottom_n: Optional[Union[int, float]] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot number of missing data points per column. Sorted by number of missing values.

    Also allows for selecting top_n/bottom_n number or percent of columns by passing an int or float

    Parameters
    ----------
    df: pd.DataFrame
        Feature DataFrame to calculate missing values from

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
    if ax is None:
        fig, ax = plt.subplots()

    missing_data = df.isna().sum().sort_values().divide(len(df)).loc[lambda x: x > 0]

    return _plot_barh(
        values=missing_data.values,
        label_names=missing_data.index,
        add_label=True,
        title="Misssing data per column",
        y_label="Feature",
        x_label="Percent Missing Data",
        ax=ax,
        top_n=top_n,
        bottom_n=bottom_n,
        is_percent=True,
        **kwargs,
    )
