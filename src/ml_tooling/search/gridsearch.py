"""
Implement Gridsearch hyperparameter optimization
"""
from typing import Iterator, List, Any

from sklearn import clone
from sklearn.model_selection import ParameterGrid

from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup
from ml_tooling.search.base import Searcher
from ml_tooling.utils import Estimator


class GridSearch(Searcher):
    """
    A Searcher implementation that uses Gridsearch to find the best set
    of hyperparameters using an exhaustive search
    """

    def prepare_gridsearch_estimators(self) -> Iterator[Estimator]:
        """
        Create an iterator over all possible estimators

        Returns
        -------
        Iterator of Estimators
        """
        grid = ParameterGrid(self.param_grid)
        yield from (clone(self.estimator).set_params(**p) for p in grid)

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        """
        Perform a gridsearch over possible hyperparameters

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
        ResultGroup
        """
        estimators = self.prepare_gridsearch_estimators()
        return self._train_estimators(
            estimators,
            metrics=metrics,
            data=data,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
        )
