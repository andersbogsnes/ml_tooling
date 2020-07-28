"""Implements RandomSearch hyperparameter optimization"""
from typing import Iterator, List, Any

import numpy as np
from sklearn import clone
from sklearn.model_selection import ParameterSampler

from ml_tooling.config import config
from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup
from ml_tooling.search.base import Searcher
from ml_tooling.utils import Estimator


class RandomSearch(Searcher):
    """
    A Searcher implementation that uses RandomSearch to find the best set
    of hyperparameters by randomly sampling the parameter space
    """

    def __init__(self, estimator: Estimator, param_grid: dict, n_iter: int = 10):
        super().__init__(estimator, param_grid)
        self.n_iter = n_iter

    def prepare_randomsearch_estimators(self) -> Iterator[Estimator]:
        """
        Prepare an iterator returning `n_iter` estimators sampling from the search space

        Returns
        -------
        Iterator of Estimators
        """
        grid = ParameterSampler(
            self.param_grid, n_iter=self.n_iter, random_state=config.RANDOM_STATE
        )

        yield from (
            clone(self.estimator).set_params(
                **{
                    k: np.array(v).item()  # To ensure compatibility with skopt Spaces
                    for k, v in p.items()
                }
            )
            for p in grid
        )

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        """
        Perform a random search over random samples from the hyperparameter space

        Parameters
        ----------
        data: Dataset
            Instance of data to train on

        metrics: List of str
            List of metrics to calculate results for

        cv: Any
            Either a CV object from sklearn or an int to specify number of folds

        n_jobs: int
            Number of jobs to calculate in parallel

        verbose: int
            Verbosity level of the method

        Returns
        -------

        """
        estimators = self.prepare_randomsearch_estimators()
        return self._train_estimators(
            estimators,
            metrics=metrics,
            data=data,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
        )
