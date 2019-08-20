from typing import List

import numpy as np
import pandas as pd

from ml_tooling.result.result import Result


class ResultGroup:
    """
    A container for results. Proxies attributes to the best result. Supports indexing like a list.
    Can output the mean score of all its results using .mean_score.
    Can convert the results to a DataFrame of results, for ease of scanning and manipulating

    """

    def __init__(self, results: List[Result]):
        self.results = sorted(results, reverse=True)

    def __getattr__(self, name):
        return getattr(self.results[0], name)

    def __dir__(self):
        proxied_dir = dir(self.results[0])
        custom_methods = ["to_dataframe", "mean_score"]
        return proxied_dir + custom_methods

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return self.results[item]

    def __repr__(self):
        results = "\n".join([str(result) for result in self.results])
        return f"[{results}]"

    def log_estimator(self, log_dir):
        for result in self.results:
            result.log_estimator(log_dir)

    def mean_score(self):
        """
        Calculates mean score across the results
        :return:
        """
        return np.mean([result.score for result in self.results])

    def to_dataframe(self, params=True) -> pd.DataFrame:
        """
        Outputs results as a DataFrame. By default, the DataFrame will contain
        all possible estimator parameters. This behaviour can be toggled using `params=False`

        :param params:
            Boolean toggling whether or not to output params as part of the DataFrame
        :return:
            pd.DataFrame of results
        """

        output = [result.to_dataframe(params) for result in self.results]

        return pd.concat(output, ignore_index=True).sort_values(
            by="score", ascending=False
        )
