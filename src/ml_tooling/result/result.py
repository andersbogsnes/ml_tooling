from functools import total_ordering

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_tooling.logging import log_results
from ml_tooling.utils import _get_estimator_name


@total_ordering
class Result:
    """
    Represents a single scoring of a estimator.
    Contains plotting methods, as well as being comparable with other results
    """

    def __init__(self,
                 estimator,
                 score,
                 viz=None,
                 metric=None,
                 ):
        self.estimator = estimator
        self.estimator_name = _get_estimator_name(estimator)
        self.score = score
        self.metric = metric
        self.plot = viz

    @property
    def estimator_params(self) -> dict:
        """
        Calls get_params on estimator. Checks if estimator is a Pipeline, in which case it
        assumes last step in pipeline is an estimator and calls get_params on that step only

        Returns
        -------
        dict
            Returns a dictionary of all params from the estimator
        """
        if isinstance(self.estimator, Pipeline):
            return self.estimator.steps[-1][1].get_params()
        return self.estimator.get_params()

    def log_estimator(self, run_dir):
        metric_score = {self.metric: float(self.score)}
        return log_results(metric_scores=metric_score,
                           estimator_name=self.estimator_name,
                           estimator_params=self.estimator_params,
                           run_dir=run_dir,
                           )

    def to_dataframe(self, params=True) -> pd.DataFrame:
        """
        Output result as a dataframe for ease of inspecting and manipulating.
        Defaults to including estimator params, which can be toggled with the params flag.
        This is useful if you're comparing different models

        Parameters
        ----------
        params: bool
            Whether or not to include estimator parameters as columns.

        Returns
        -------
        pd.DataFrame
            DataFrame of the result
        """
        estimator_params_dict = {}
        if params:
            estimator_params_dict = self.estimator_params

        estimator_params_dict['score'] = self.score
        estimator_params_dict['metric'] = self.metric

        return pd.DataFrame([estimator_params_dict])

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"<Result {self.estimator_name}: " \
            f"{self.metric}: {np.round(self.score, 2)} >"
