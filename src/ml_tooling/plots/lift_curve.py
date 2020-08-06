"""
Plot a lift curve, which shows the gain of the model compared to a random guess.
Implements OneVsRest for multi-class classifications.

Inspired by https://www3.nd.edu/~busiforc/Lift_chart.html
"""

from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import label_binarize

from ml_tooling.metrics import lift_score
from ml_tooling.utils import VizError, DataType


def plot_lift_curve(
    y_true: DataType,
    y_proba: DataType,
    title: str = None,
    ax: Axes = None,
    labels: List[str] = None,
    threshold: float = 0.5,
) -> Axes:
    """
    Plot a lift chart from results. Also calculates lift score based on a .5 threshold

    Parameters
    ----------
    y_true: DataType
        True labels

    y_proba: DataType
        Model's predicted probability

    title: str
        Plot title

    ax: Axes
        Pass your own ax

    labels: List of str
        Labels to use per class

    threshold: float
        Threshold to use when determining lift score

    Returns
    -------
    matplotlib.Axes
    """

    if ax is None:
        fig, ax = plt.subplots()

    title = "Lift Curve" if title is None else title
    classes = np.unique(y_true)
    binarized_labels = label_binarize(y_true, classes=classes)

    if labels and len(labels) != len(classes):
        raise VizError(
            "Number of labels must match number of classes: "
            f"got {len(labels)} labels and {len(classes)} classes"
        )

    if binarized_labels.shape[1] == 1:
        # Binary classification case
        percents, gains = _cum_gain_curve(binarized_labels, y_proba[:, 1])
        score = lift_score(binarized_labels.ravel(), y_proba[:, 1] > threshold)
        ax.plot(percents, gains / percents, label=f"$Lift = {score:.2f}$")
    else:
        # Multi-class case
        for class_ in classes:
            percents, gains = _cum_gain_curve(
                binarized_labels[:, class_], y_proba[:, class_]
            )
            score = lift_score(
                binarized_labels[:, class_], y_proba[:, class_] > threshold
            )
            ax.plot(
                percents,
                gains / percents,
                label=f"Class {labels[class_] if labels else class_} "
                f"$Lift = {score:.2f}$ ",
            )

    ax.axhline(y=1, color="grey", linestyle="--", label="Baseline")
    ax.set_title(title)
    ax.set_ylabel("Lift")
    ax.set_xlabel("% of Data")
    formatter = PercentFormatter(xmax=1)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend()
    return ax


def _cum_gain_curve(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a cumulative gain curve of how many positives are captured
    per percent of sorted data.

    :param y_true:
        True labels

    :param y_proba:
        Predicted label

    :return:
        array of data percents and cumulative gain
    """
    n = len(y_true)
    n_true = np.sum(y_true)

    idx = np.argsort(y_proba)[::-1]  # Reverse sort to get descending values
    cum_gains = np.cumsum(y_true[idx]) / n_true
    percents = np.arange(1, n + 1) / n
    return percents, cum_gains
