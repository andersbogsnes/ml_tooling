import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import r2_score

from ml_tooling.utils import DataType


def plot_prediction_error(y_true: DataType,
                          y_pred: DataType,
                          title: str = None,
                          ax: Axes = None) -> Axes:
    """
    Plots prediction error of regression model

    :param y_true:
        True values

    :param y_pred:
        Model's predicted values

    :param title:
        Plot title

    :param ax:
        Pass your own ax

    :return:
        matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    title = f"Prediction Error" if title is None else title

    r2 = r2_score(y_true, y_pred)

    ax.scatter(y_true, y_pred, label=f"$R^2 = {r2}$")
    ax.set_ylabel('$\hat{y}$')
    ax.set_xlabel('$y$')
    ax.set_title(title)
    min_x, min_y = np.min(y_true), np.min(y_pred)
    max_x, max_y = np.max(y_true), np.max(y_pred)
    ax.plot((min_x, max_x), (min_y, max_y), c='grey', linestyle='--')
    return ax