from functools import total_ordering

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_tooling.logging import log_model
from ml_tooling.utils import _get_model_name


@total_ordering
class Result:
    """
    Represents a single scoring of a model.
    Contains plotting methods, as well as being comparable with other results
    """

    def __init__(self,
                 model,
                 score,
                 viz=None,
                 metric=None,
                 ):
        self.model = model
        self.model_name = _get_model_name(model)
        self.score = score
        self.metric = metric
        self.plot = viz

    @property
    def model_params(self) -> dict:
        """
        Calls get_params on estimator. Checks if estimator is a Pipeline, in which case it
        assumes last step in pipeline is an estimator and calls get_params on that step only

        Returns
        -------
        dict
            Returns a dictionary of all params from the model
        """
        if isinstance(self.model, Pipeline):
            return self.model.steps[-1][1].get_params()
        return self.model.get_params()

    def log_model(self, run_dir):
        metric_score = {self.metric: float(self.score)}
        return log_model(metric_scores=metric_score,
                         model_name=self.model_name,
                         model_params=self.model_params,
                         run_dir=run_dir,
                         )

    def to_dataframe(self, params=True) -> pd.DataFrame:
        """
        Output result as a dataframe for ease of inspecting and manipulating.
        Defaults to including model params, which can be toggled with the params flag.
        This is useful if you're comparing different models

        Parameters
        ----------
        params: bool
            Whether or not to include model parameters as columns.

        Returns
        -------
        pd.DataFrame
            DataFrame of the result
        """
        model_params_dict = {}
        if params:
            model_params_dict = self.model_params

        model_params_dict['score'] = self.score
        model_params_dict['metric'] = self.metric

        return pd.DataFrame([model_params_dict])

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"<Result {self.model_name}: " \
            f"{self.metric}: {np.round(self.score, 2)} >"
