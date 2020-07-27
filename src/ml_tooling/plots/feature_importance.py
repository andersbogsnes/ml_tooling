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
    estimator.fit(x, y)

    trained_estimator: BaseEstimator = estimator.steps[-1][1] if is_pipeline(
        estimator
    ) else estimator

    if hasattr(trained_estimator, "coef_"):
        feature_importances: np.ndarray = trained_estimator.coef_
        x_label = "Coefficients"

    elif hasattr(trained_estimator, "feature_importances_"):
        feature_importances: np.ndarray = trained_estimator.feature_importances_
        x_label = "Feature Importances"
    else:
        raise MLToolingError(
            "Estimator must have one of coef_ or feature_importances_."
            f"{estimator} has neither."
        )

    labels = _get_labels_from_pipeline(estimator, x)
    plot_title = title if title else "Feature Importances"
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
