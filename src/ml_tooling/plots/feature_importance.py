from typing import Union

import pandas as pd
from matplotlib.axes import Axes

from sklearn.inspection import permutation_importance
from ml_tooling.plots.utils import _plot_barh
from ml_tooling.utils import DataType, Estimator
from sklearn.base import is_classifier


def plot_feature_importance(
    estimator: Estimator,
    x: pd.DataFrame,
    y: DataType,
    scoring="default",
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: int = None,
    ax: Axes = None,
    bottom_n: Union[int, float] = None,
    top_n: Union[int, float] = None,
    add_label: bool = True,
    title: str = "",
    **kwargs,
) -> Axes:
    if scoring == "default":
        scoring = "accuracy" if is_classifier(estimator) else "r2"

    importances = permutation_importance(
        estimator,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    feature_importance = importances.importances_mean
    labels = x.columns
    plot_title = title if title else f"Feature Importances ({scoring.title()})"
    ax = _plot_barh(
        feature_importance,
        labels,
        add_label=add_label,
        title=plot_title,
        x_label=f"Permuted Feature Importance ({scoring.title()}) Relative to Baseline",
        y_label="Feature Labels",
        ax=ax,
        top_n=top_n,
        bottom_n=bottom_n,
        **kwargs,
    )
    return ax
