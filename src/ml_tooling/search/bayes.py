from typing import List, Any

import sklearn.utils.fixes

# Hack to fix https://github.com/scikit-optimize/scikit-optimize/issues/902
from numpy.ma import MaskedArray

sklearn.utils.fixes.MaskedArray = MaskedArray

from sklearn import clone
from skopt import Optimizer
from skopt.utils import dimensions_aslist, point_asdict

import numpy as np

from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup, Result
from ml_tooling.search.base import Searcher
from ml_tooling.utils import Estimator


class BayesSearch(Searcher):
    def __init__(self, estimator: Estimator, param_grid: dict, n_iter):
        super().__init__(estimator, param_grid)
        self.n_iter = n_iter

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        optimizer = Optimizer(dimensions_aslist(self.param_grid))
        results = [
            self._step(optimizer, data, metrics, cv, n_jobs, verbose)
            for _ in range(self.n_iter)
        ]
        return ResultGroup(results).sort()

    def _step(
        self,
        optimizer: Optimizer,
        data: Dataset,
        metrics: List[str],
        cv: Any,
        n_jobs: int,
        verbose: int,
    ) -> Result:
        params = optimizer.ask(1)
        params = [np.array(p).item() for p in params]

        # make lists into dictionaries
        params_dict = point_asdict(self.param_grid, params)
        estimator = clone(self.estimator).set_params(**params_dict)
        result = Result.from_estimator(
            estimator=estimator,
            data=data,
            metrics=metrics,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        optimizer.tell([params], [-result.metrics[0].score])
        return result
