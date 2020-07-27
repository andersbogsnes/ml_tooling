from typing import Iterator, List, Any

from sklearn import clone
from sklearn.model_selection import ParameterSampler

from ml_tooling.config import config
from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup
from ml_tooling.search.base import Searcher
from ml_tooling.utils import Estimator


class RandomSearch(Searcher):
    def __init__(self, estimator: Estimator, param_grid: dict, n_iter: int = 10):
        super().__init__(estimator, param_grid)
        self.n_iter = n_iter

    def prepare_randomsearch_estimators(self) -> Iterator[Estimator]:
        grid = ParameterSampler(
            self.param_grid, n_iter=self.n_iter, random_state=config.RANDOM_STATE
        )
        yield from (clone(self.estimator).set_params(**p) for p in grid)

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        estimators = self.prepare_randomsearch_estimators()
        return self._train_estimators(
            estimators,
            metrics=metrics,
            data=data,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
        )
