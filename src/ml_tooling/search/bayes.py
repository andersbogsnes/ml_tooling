"""
Implements Bayesian Hyperparameter optimization
"""

from typing import List, Any
import logging

import numpy as np

from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup, Result
from ml_tooling.search.base import Searcher
from ml_tooling.utils import Estimator

import sklearn.utils.fixes
from sklearn import clone

# Hack to fix https://github.com/scikit-optimize/scikit-optimize/issues/902
# TODO: Change this when issue is solved
from numpy.ma import MaskedArray

sklearn.utils.fixes.MaskedArray = MaskedArray

from skopt import Optimizer  # noqa: 402
from skopt.utils import dimensions_aslist, point_asdict  # noqa: 402

logger = logging.getLogger(__name__)


class BayesSearch(Searcher):
    """
    A Searcher implementation that uses Bayesian optimization to find the best set
    of hyperparameters given an n_iter calculation budget
    """

    def __init__(self, estimator: Estimator, param_grid: dict, n_iter: int):
        """
        Parameters
        ----------
        estimator: Estimator
            Estimator to optimize

        param_grid: dict
            Dictionary of parameters to search over

        n_iter: int
            Number of iterations to perform
        """
        super().__init__(estimator, param_grid)
        self.n_iter = n_iter

    def search(
        self, data: Dataset, metrics: List[str], cv: Any, n_jobs: int, verbose: int = 0
    ) -> ResultGroup:
        """
        Perform a bayesian search over the specified hyperparameters

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

        optimizer = Optimizer(dimensions_aslist(self.param_grid))
        logger.info("Starting Bayesian search...")
        results = [
            self._step(optimizer, data, metrics, cv, n_jobs, verbose)
            for _ in range(self.n_iter)
        ]
        logger.info("Finished Bayesian search...")
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
        """
        Performs a step in the Bayesian optimization

        Parameters
        ----------
        optimizer: Optimizer
            An instance of skopt's Optimizer

        data: Dataset
           Instance of data to train on

        metrics: List of str
            List of metrics to calculate results for

        cv: Any
            Either a CV object from sklearn or an int to specify number of folds

        n_jobs
            Number of jobs to calculate in parallel

        verbose
            Verbosity level of the method

        Returns
        -------
        Result
        """
        params = optimizer.ask()
        params = [np.array(p).item() for p in params]

        # make lists into dictionaries
        params_dict = point_asdict(self.param_grid, params)
        estimator = clone(self.estimator).set_params(**params_dict)
        logger.info("Fitting estimator...")
        logger.debug("Fitting estimator %s", estimator)

        result = Result.from_estimator(
            estimator=estimator,
            data=data,
            metrics=metrics,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        logger.info("Result: %s", result)
        optimizer.tell([params], [-result.metrics[0].score])
        return result
