"""Low-level implementation of plotting feature importance from an estimator"""

from typing import Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator

from ml_tooling.plots.utils import _plot_barh
from ml_tooling.utils import (
    Estimator,
    is_pipeline,
    _get_labels_from_pipeline,
    VizError,
)


def plot_feature_importance(
    estimator: Estimator,
    x: pd.DataFrame,
    ax: Axes = None,
    class_index: int = None,
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

    ax: Axes
        Matplotlib axes to draw the graph on. Creates a new one by default

    class_index: int, optional
        In a multi-class setting, choose which class to get feature importances for. If None,
        will assume a binary classifier

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

    trained_estimator: BaseEstimator = estimator.steps[-1][1] if is_pipeline(
        estimator
    ) else estimator

    default_class_name = 0 if class_index is None else class_index

    if hasattr(trained_estimator, "coef_"):
        feature_importances: np.ndarray = np.atleast_2d(getattr(trained_estimator, "coef_"))
        try:
            feature_importances = feature_importances[default_class_name, :]
        except IndexError:
            raise VizError(f"Tried to get coefficients for {class_index} - "
                           f"class_index can only be one of "
                           f"{[x for x in range(feature_importances.shape[0])]}")

        x_label = "Coefficients"

    elif hasattr(trained_estimator, "feature_importances_"):
        feature_importances: np.ndarray = getattr(
            trained_estimator, "feature_importances_"
        )
        x_label = "Feature Importances"

    else:
        raise VizError(
            "Estimator must have one of coef_ or feature_importances_."
            f"{estimator} has neither. Make sure that it has been fitted"
        )

    labels = _get_labels_from_pipeline(estimator, x)

    default_title = (
        "Feature Importances"
        if class_index is None
        else f"Feature Importances - Class {default_class_name}"
    )

    plot_title = title if title else default_title

    ax = _plot_barh(
        feature_importances,
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
