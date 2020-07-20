import itertools
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import ml_tooling.metrics
from ml_tooling.utils import DataType


def plot_confusion_matrix(
    y_true: DataType,
    y_pred: DataType,
    normalized: bool = True,
    title: str = None,
    ax: Axes = None,
    labels: Sequence[str] = None,
) -> Axes:
    """
    Plots a confusion matrix of predicted labels vs actual labels

    :param y_true:
        True labels

    :param y_pred:
        Predicted labels from estimator

    :param normalized:
        Whether to normalize counts in matrix

    :param title:
        Title for plot

    :param ax:
        Pass your own ax

    :param labels:
        Pass custom list of labels

    :return:
        matplotlib.Axes
    """

    title = "Confusion Matrix" if title is None else title

    if normalized:
        title = f"{title} - Normalized"

    cm = ml_tooling.metrics.confusion_matrix(y_true, y_pred, normalized=normalized)

    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        unique_labels = np.unique(y_true)
        labels = list(unique_labels)

    cax = ax.matshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticklabels(labels)
    ax.xaxis.set_ticks_position("bottom")

    plt.colorbar(cax, ax=ax)
    fmt = ".2f" if normalized else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    return ax
