"""Low-level implementation of plotting feature importance from an estimator"""

from typing import Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator

from ml_tooling.plots.utils import _plot_barh
from ml_tooling.utils import (
    DataType,
    Estimator,
    is_pipeline,
    _get_labels_from_pipeline,
    MLToolingError,
)


def plot_feature_importance(
    estimator: Estimator,
    x: pd.DataFrame,
    y: DataType,
    ax: Axes = None,
    bottom_n: Union[int, float] = None,
    top_n: Union[int, float] = None,
    add_label: bool = True,
    title: str = "",
    **kwargs,
) -> Axes:
    """
    Plot either the estimator coefficients or the estimator
    feature importances depending on what is provided by the estimator.

    see also :func:ml_tooling.plot.plot_permutation_importance for an
    unbiased version of feature importance using permutation importance


    Parameters
    ----------
    estimator: Estimator
        Estimator to use to calculate permuted feature importance

    x: DataType
        Features to calculate permuted feature importance for

    y: DataType
        Target to use in scoring

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
    estimator.fit(x, y)

    trained_estimator: BaseEstimator = estimator.steps[-1][1] if is_pipeline(
        estimator
    ) else estimator

    if hasattr(trained_estimator, "coef_"):
        feature_importances: np.ndarray = getattr(trained_estimator, "coef_").squeeze()
        x_label = "Coefficients"

    elif hasattr(trained_estimator, "feature_importances_"):
        feature_importances: np.ndarray = getattr(
            trained_estimator, "feature_importances_"
        )
        x_label = "Feature Importances"

    else:
        raise MLToolingError(
            "Estimator must have one of coef_ or feature_importances_."
            f"{estimator} has neither."
        )

    labels = _get_labels_from_pipeline(estimator, x)

    plot_title = title if title else "Feature Importances"

    ax = _plot_barh(
        feature_importances.squeeze(),
        labels,
        add_label=add_label,
        title=plot_title,
        x_label=x_label,
        y_label="Feature Labels",
        ax=ax,
        top_n=top_n,
        bottom_n=bottom_n,
        **kwargs,
    )
    return ax
