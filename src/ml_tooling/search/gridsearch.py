from typing import Iterator, List, Any

from sklearn import clone
from sklearn.model_selection import ParameterGrid

from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup
from ml_tooling.search.base import Searcher
from ml_tooling.utils import Estimator


class GridSearch(Searcher):
    def prepare_gridsearch_estimators(self) -> Iterator[Estimator]:
        grid = ParameterGrid(self.param_grid)
        yield from (clone(self.estimator).set_params(**p) for p in grid)

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        estimators = self.prepare_gridsearch_estimators()
        return self._train_estimators(
            estimators,
            metrics=metrics,
            data=data,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
        )
