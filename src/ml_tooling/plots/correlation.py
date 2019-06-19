import pandas as pd
from matplotlib.axes import Axes

from ml_tooling.metrics.correlation import target_correlation
from ml_tooling.plots.feature_importance import plot_feature_importance
from ml_tooling.utils import DataType


def plot_target_correlation(features: pd.DataFrame,
                            target: DataType,
                            method='spearman',
                            ax: Axes = None,
                            top_n=None,
                            bottom_n=None) -> Axes:
    correlation = target_correlation(features, target, method=method, ascending=True)
    ax = plot_feature_importance(correlation.values,
                                 correlation.index,
                                 values=True,
                                 title="Feature to Target Correlation",
                                 x_label=f"{method.title()} Correlation",
                                 ax=ax,
                                 top_n=top_n,
                                 bottom_n=bottom_n
                                 )
    return ax
