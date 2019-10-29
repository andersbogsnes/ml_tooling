import pathlib
from typing import List

from ml_tooling.result.result import Result
import attr


@attr.s(auto_attribs=True)
class ResultGroup:
    """
    A container for results. Proxies attributes to the best result. Supports indexing like a list.
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
            by = self.results[0].metrics[0].name

        scores = [
            (i, metric.score)
            for i, result in enumerate(self.results)
            for metric in result.metrics
            if by == metric.name
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        self.results = [self.results[i] for i, _ in scores]
        return self

    def log(self, log_dir: pathlib.Path):
        for result in self.results:
            result.log(savedir=log_dir)
