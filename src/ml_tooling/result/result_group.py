import pathlib
from typing import List

from ml_tooling.logging import Log
from ml_tooling.result.result import Result
import attr


@attr.s(auto_attribs=True)
class ResultGroup:
    """
    A container for results. Proxies attributes to the best result. Supports indexing like a list.
    Can output the mean score of all its results using .mean_score.
    Can convert the results to a DataFrame of results, for ease of scanning and manipulating

    """

    results: List[Result]

    def __getattr__(self, name):
        return getattr(self.results[0], name)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return self.results[item]

    def sort(self, by=None):
        if by is None:
            by = self.results[0].metrics[0].metric

        scores = [
            (i, metric.score)
            for i, result in enumerate(self.results)
            for metric in result.metrics
            if by == metric.metric
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        self.results = [self.results[i] for i, _ in scores]
        return self

    def log_estimator(self, log_dir: pathlib.Path):
        for result in self.results:
            log = Log.from_result(result)
            log.save_log(log_dir)
