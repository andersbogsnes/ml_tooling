from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import precision_recall_curve, average_precision_score

from ml_tooling.utils import DataType


def plot_pr_curve(
    y_true: DataType, y_proba: DataType, title: str = None, ax: Axes = None
) -> Axes:
    """   Plot precision-recall curve. Works only with probabilities.

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

    Returns
    -------
    matplotlib.Axes
        Plot of precision-recall curve
    """

    title = "Precision-Recall curve" if title is None else title

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    score = average_precision_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots()

    ax.step(
        recall, precision, label=f"Average precision: {score}", alpha=0.95, where="post"
    )
    ax.fill_between(recall, precision, alpha=0.2, step="post")
    ax.set_title(title)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.set_xlim(left=0.0, right=1.0)
    ax.legend(loc="best")
    plt.tight_layout()
    return ax
