from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from ml_tooling.utils import DataType, VizError


def plot_pr_curve(
    y_true: DataType,
    y_proba: DataType,
    title: str = None,
    ax: Axes = None,
    labels: List[str] = None,
) -> Axes:
    """
    Plot precision-recall curve. Works only with probabilities.

    Parameters
    ----------
    y_true : DataType
        True labels

    y_proba : DataType
        Probability estimate from estimator

    title : str
        Plot title

    ax : plt.Axes
        Pass in your own ax

    labels: List of str, optional
        Labels for each class

    Returns
    -------
    plt.Axes
        Plot of precision-recall curve
    """

    if ax is None:
        fig, ax = plt.subplots()

    title = "Precision-Recall curve" if title is None else title
    classes = np.unique(y_true)
    binarized_labels = label_binarize(y_true, classes=classes)

    if labels and len(labels) != len(classes):
        raise VizError(
            "Number of labels must match number of classes:"
            f"got {len(labels)} labels and {len(classes)} classes"
        )

    plot_kwargs = dict(alpha=0.95, where="post")

    if binarized_labels.shape[1] == 1:
        # Binary classification case
        precision, recall, _ = precision_recall_curve(binarized_labels, y_proba[:, 1])
        precision_score = average_precision_score(binarized_labels, y_proba[:, 1])
        ax.step(
            recall,
            precision,
            label=f"Average precision score: {precision_score:.2f}",
            **plot_kwargs,
        )
    else:
        # Multi-class case
        for class_ in classes:
            precision, recall, _ = precision_recall_curve(
                binarized_labels[:, class_], y_proba[:, class_]
            )
            precision_score = average_precision_score(
                binarized_labels[:, class_], y_proba[:, class_]
            )
            ax.step(
                recall,
                precision,
                label=f"Class {labels[class_] if labels else class_} "
                f"- Average precision score: {precision_score:.2f}",
                **plot_kwargs,
            )

    ax.set_title(title)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.set_xlim(left=0.0, right=1.0)
    ax.legend(loc="best")
    plt.tight_layout()
    return ax
