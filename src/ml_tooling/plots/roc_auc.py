from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve, roc_auc_score

from ml_tooling.utils import DataType


def plot_roc_auc(y_true: DataType, y_proba: DataType, title: str = None, ax: Axes = None) -> Axes:
    """
    Plot ROC AUC curve. Works only with probabilities

    :param y_true:
        True labels

    :param y_proba:
        Probability estimate from model

    :param title:
        Plot title

    :param ax:
        Pass in your own ax

    :return:
        matplotlib.Axes
    """
    title = 'ROC AUC curve' if title is None else title

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    score = roc_auc_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"ROC Score: {score}")
    ax.plot([0, 1], '--')
    ax.set_title(title)
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc='best')
    plt.tight_layout()
    return ax