import numpy as np
from matplotlib import pyplot as plt

from ..metrics import lift_score
from ..metrics.utils import _cum_gain_curve
from .utils import VizError


def plot_lift_curve(y_true, y_proba, title=None, ax=None):
    """
    Plot a lift chart from results. Also calculates lift score based on a .5 threshold

    :param y_true:
        True labels

    :param y_proba:
        Model's predicted probability

    :param title:
        Plot title

    :param ax:
        Pass your own ax

    :return:
        matplotlib.Axes
    """

    if y_proba.ndim > 1:
        raise VizError("Only works in binary classification. Pass a 1d list")

    if ax is None:
        fig, ax = plt.subplots()

    title = "Lift Curve" if title is None else title

    # Get numpy array as _cum_gain_curve takes numpy arrays
    y_true = np.array(y_true)

    percents, gains = _cum_gain_curve(y_true, y_proba)
    positives = np.where(y_proba > 0.5, 1, 0)
    score = lift_score(y_true, positives)

    ax.plot(percents, gains / percents, label=f"$Lift = {score:.2f}$ ")
    ax.axhline(y=1, color="grey", linestyle="--", label="Baseline")
    ax.set_title(title)
    ax.set_ylabel("Lift")
    ax.set_xlabel("% of Data")
    ax.legend()
    return ax
