from typing import Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from ml_tooling.plots import plot_target_correlation
from ml_tooling.config import MPL_STYLESHEET


class DataVisualize:
    def __init__(self, data):
        self.data = data

    def target_correlation(
        self,
        method: str = "spearman",
        ax: Optional[Axes] = None,
        top_n: Optional[int] = None,
        bottom_n: Optional[int] = None,
    ) -> Axes:
        """
        Plot the correlation between each feature
        and the target variable using the given value.

        Also allows selecting how many features to show by
        setting the top_n and/or bottom_n parameters.

        Parameters
        ----------

        method: str
            Which method to use when calculating
            correlation. Supports one of 'pearson', 'spearman', 'kendall'.

        ax: plt.Axes
            Matplotlib axes to draw the graph on. Creates a new one by default

        top_n: int, float
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        bottom_n: int, float
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        Returns
        -------
        plt.Axes
        """
        with plt.style.context(MPL_STYLESHEET):
            return plot_target_correlation(
                features=self.data.x,
                target=self.data.y,
                method=method,
                ax=ax,
                top_n=top_n,
                bottom_n=bottom_n,
            )
