from functools import total_ordering

import numpy as np
import pandas as pd

from ml_tooling.logging.log_estimator import create_log


@total_ordering
class Result:
    """
    Represents a single scoring of a estimator.
    Contains plotting methods, as well as being comparable with other results
    """

    def __init__(self, model, data, score, metric=None):
        self.model = model
        self.score = score
        self.metric = metric
        self.data = data
        self.plot = model._setup_plotter(data)

    def dump(self, saved_estimator_path=None):
        metric_score = {self.metric: float(self.score)}
        name = f"{self.data.class_name}_{self.model.estimator_name}"

        return create_log(
            name=name,
            metric_scores=metric_score,
            serialized_estimator=self.model.dump(),
            saved_estimator_path=saved_estimator_path,
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
            estimator_params_dict = self.model.estimator.get_params()

        estimator_params_dict["score"] = self.score
        estimator_params_dict["metric"] = self.metric

        return pd.DataFrame([estimator_params_dict])

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return (
            f"<Result {self.model.estimator_name}: "
            f"{self.metric}: {np.round(self.score, 2)}>"
        )
