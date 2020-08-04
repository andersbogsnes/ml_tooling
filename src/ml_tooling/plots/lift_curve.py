"""
Plot a lift curve, which shows the gain of the model compared to a random guess.
Implements OneVsRest for multi-class classifications.

Inspired by https://www3.nd.edu/~busiforc/Lift_chart.html
"""

from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ml_tooling.metrics import lift_score
from ml_tooling.utils import VizError, DataType


def plot_lift_curve(y_true: DataType,
                    y_proba: DataType,
                    title: str = None,
                    ax: Axes = None,
                    labels: List[str] = None) -> Axes:
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

    Returns
    -------
    matplotlib.Axes
    """

    if ax is None:
        fig, ax = plt.subplots()

    title = "Lift Curve" if title is None else title

    # Convert to numpy array as _cum_gain_curve takes numpy arrays
    y_true = np.array(y_true)

    classes = np.unique(y_true)

    if labels and len(classes) != len(labels):
        raise VizError(f"Number of labels must equal number of classes: got {len(classes)} classes"
                       f" and {len(labels)} labels")

    for class_label in classes:
        if len(classes) == 2 and class_label == 0:
            # Skip label 0 in the binary classification case
            continue

        target = y_true == class_label
        percents, gains = _cum_gain_curve(target, y_proba[:, class_label])
        positives = np.where(y_proba[:, class_label] > 0.5, 1, 0)
        score = lift_score(target, positives)
        ax.plot(percents,
                gains / percents,
                label=f"Class {labels[class_label] if labels else class_label} "
                      f"$Lift = {score:.2f}$ ")

    ax.axhline(y=1, color="grey", linestyle="--", label="Baseline")
    ax.set_title(title)
    ax.set_ylabel("Lift")
    ax.set_xlabel("% of Data")
    ax.legend()
    return ax


def _cum_gain_curve(
        y_true: np.ndarray, y_proba: np.ndarray, positive_label=1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a cumulative gain curve of how many positives are captured
    per percent of sorted data.

    :param y_true:
        True labels

    :param y_proba:
        Predicted label

    :param positive_label:
        Which class is considered positive class in multi-class settings

    :return:
        array of data percents and cumulative gain
    """
    n = len(y_true)
    n_true = np.sum(y_true == positive_label)

    idx = np.argsort(y_proba)[::-1]  # Reverse sort to get descending values
    cum_gains = np.cumsum(y_true[idx]) / n_true
    percents = np.arange(1, n + 1) / n
    return percents, cum_gains
