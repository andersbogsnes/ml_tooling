from typing import List, Any, Iterable

from ml_tooling.data import Dataset
from ml_tooling.metrics import Metrics
from ml_tooling.result import ResultGroup, Result
from ml_tooling.utils import Estimator


class Searcher:
    def __init__(self, estimator: Estimator, param_grid: dict):
        self.estimator = estimator
        self.param_grid = param_grid

    @staticmethod
    def _train_estimators(
        estimators: Iterable[Estimator],
        metrics: List[str],
        data: Dataset,
        n_jobs: int,
        cv: Any,
        verbose: int,
    ):
        metrics = Metrics.from_list(metrics)
        results = [
            Result.from_estimator(
                estimator=estimator,
                metrics=metrics,
                data=data,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            for estimator in estimators
        ]

        return ResultGroup(results).sort()

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        raise NotImplementedError
