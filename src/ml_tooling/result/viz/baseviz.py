from typing import Union

from matplotlib import pyplot as plt

from ml_tooling.metrics.permutation_importance import _get_feature_importance
from ml_tooling.plots import plot_feature_importance
from ml_tooling.utils import _get_model_name


class BaseVisualize:
    """
    Base class for visualizers
    """

    def __init__(self, model, config, data):
        self._model = model
        self._model_name = _get_model_name(model)
        self._config = config
        self._data = data

    @property
    def default_metric(self):
        """
        Finds estimator_type for estimator in a BaseVisualize and returns default
        metric for this class stated in .config. If passed estimator is a Pipeline,
        assume last step is the estimator.

        Returns
        -------
        str
            Name of the metric

        """

        return self._config.CLASSIFIER_METRIC if self._model._estimator_type == 'classifier' \
            else self._config.REGRESSION_METRIC

    def feature_importance(self,
                           samples,
                           values: bool = True,
                           top_n: Union[int, float] = None,
                           bottom_n: Union[int, float] = None,
                           n_jobs=None,
                           **kwargs) -> plt.Axes:
        """
        Visualizes feature importance of the model through permutation.

        Parameters
        ----------
        samples : None, int, float

            None - Original data set i used. Not recommended for small data sets

            float - A new smaller data set is made from resampling with
                replacement form the original data set. Not recommended for small data sets.
                Recommended for very large data sets.

            Int - A new  data set is made from resampling with replacement form the original data.
                samples sets the number of resamples. Recommended for small data sets
                to ensure stable estimates of feature importance.

        values : bool
            Toggles value labels on end of each bar

        top_n: int, float
            If top_n is an integer, return top_n features.
            If top_n is a float between (0, 1), return top_n percent features

        bottom_n: int, float
            If bottom_n is an integer, return bottom_n features.
            If bottom_n is a float between (0, 1), return bottom_n percent features

        n_jobs: int
            Overwrites N_JOBS from settings. Useful if data is to big to fit
            in memory multiple times.

        kwargs

        Returns
        -------
            matplotlib.Axes
        """

        n_jobs = self._config.N_JOBS if n_jobs is None else n_jobs
        title = f"Feature Importance - {self._model_name}"
        importance, baseline = _get_feature_importance(self,
                                                       samples=samples,
                                                       seed=self._config.RANDOM_STATE,
                                                       n_jobs=n_jobs,
                                                       verbose=self._config.VERBOSITY)
        labels = self._data.train_x.columns
        x_label = f"Importance:  Decrease in {self.default_metric} from baseline of {baseline}"

        with plt.style.context(self._config.STYLE_SHEET):
            return plot_feature_importance(importance,
                                           labels,
                                           values=values,
                                           title=title,
                                           x_label=x_label,
                                           top_n=top_n,
                                           bottom_n=bottom_n,
                                           **kwargs)