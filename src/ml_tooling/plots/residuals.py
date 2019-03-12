from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import r2_score

from ml_tooling.utils import DataType


def plot_residuals(y_true: DataType,
                   y_pred: DataType,
                   title: str = None,
                   ax: Axes = None) -> Axes:
    """
    Plots residuals from a regression.

    :param y_true:
        True values

    :param y_pred:
        Models predicted value

    :param title:
        Plot title

    :param ax:
        Pass your own ax

    :return:
        matplotlib.Axes
    """
    title = f'Residual Plot' if title is None else title

    residuals = y_pred - y_true
    r2 = r2_score(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(y_pred, residuals, label=f'$R^2 = {r2:0.3f}$')
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Predicted Value')
    ax.set_title(title)
    ax.legend(loc='best')
    return ax