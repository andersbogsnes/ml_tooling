from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

from ml_tooling.utils import DataType, VizError
import numpy as np

def plot_roc_auc(y_true: DataType,
                 y_proba: DataType,
                 title: str = None,
                 ax: Axes = None,
                 labels: List[str] = None
) -> Axes:
    """
    Plot ROC AUC curve. Works only with probabilities

    Parameters
    ----------
    y_true: DataType
        True labels

    y_proba: DataType
        Probability estimate from estimator

    title: str
        Plot title

    ax: Axes
        Pass in your own ax

    labels: List of str
        Optionally specify label names

    Returns
    -------
    plt.Axes
        Plot of ROC AUC curve
    """

    if ax is None:
        fig, ax = plt.subplots()

    title = "ROC AUC curve" if title is None else title
    classes = np.unique(y_true)
    binarized_labels = label_binarize(y_true, classes=classes)

    if labels and len(labels) != len(classes):
        raise VizError(
            "Number of labels must match number of classes:"
            f"got {len(labels)} labels and {len(classes)} classes"
        )

    if binarized_labels.shape[1] == 1:
        # Binary classification case

        fpr, tpr, _ = roc_curve(binarized_labels, y_proba[:, 1])
        score = roc_auc_score(binarized_labels, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"ROC Score: {score:.2f}")

    else:
        # Multi-class case
        for class_ in classes:
            fpr, tpr, _ = roc_curve(binarized_labels[:, class_], y_proba[:, class_])
            score = roc_auc_score(binarized_labels[:, class_], y_proba[:, class_])
            ax.plot(fpr, tpr, label=f"Class {labels[class_] if labels else class_} - "
                                    f"ROC Score: {score:.2f}")

    ax.plot([0, 1], "--")
    ax.set_title(title)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend(loc="best")
    plt.tight_layout()
    return ax
