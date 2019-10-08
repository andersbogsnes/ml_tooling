from typing import Union, List

import attr
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


@attr.s
class Metric:
    metric: str = attr.ib()
    score: Union[float, int] = attr.ib(default=None)
    cross_val_scores: np.ndarray = attr.ib(default=None)

    def score_metric(self, estimator, x, y):
        scoring_func = get_scorer(self.metric)
        self.score = scoring_func(estimator, x, y)

    @property
    def std(self):
        if self.cross_val_scores:
            return np.std(self.cross_val_scores)

    def score_metric_cv(self, estimator, x, y, cv, n_jobs, verbose):
        self.cross_val_scores = cross_val_score(
            estimator, x, y, cv=cv, scoring=self.metric, n_jobs=n_jobs, verbose=verbose
        )
        self.score = float(np.mean(self.cross_val_scores))


@attr.s(auto_attribs=True)
class Metrics:
    metrics: List[Metric]

    @classmethod
    def from_list(cls, metrics: List[str]):
        return cls([Metric(metric=metric) for metric in metrics])

    def list_metrics(self):
        return [m.metric for m in self.metrics]

    def dump(self):
        return {m.metric: float(m.score) for m in self.metrics}

    def score_estimator(self, estimator, x, y):
        for metric in self.metrics:
            metric.score_metric(estimator, x, y)

    def score_estimator_cv(self, estimator, x, y, cv, n_jobs, verbose):
        for metric in self.metrics:
            metric.score_metric_cv(estimator, x, y, cv, n_jobs, verbose)

    def __len__(self):
        return len(self.metrics)

    def __getattr__(self, item):
        if hasattr(self.metrics[0], item):
            return getattr(self.metrics[0], item)

    def __getitem__(self, item):
        return self.metrics[item]

    def __contains__(self, item):
        for metric in self.metrics:
            if metric.metric == item:
                return True
        return False
