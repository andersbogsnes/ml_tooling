"""
Base implementation for Hyperparameter optimization
"""

from typing import List, Any, Iterable
import logging

from ml_tooling.data import Dataset
from ml_tooling.result import ResultGroup, Result
from ml_tooling.utils import Estimator

logger = logging.getLogger("ml_tooling.search")


class Searcher:
    """Base class for Hyperparameter optimization"""

    def __init__(self, estimator: Estimator, param_grid: dict):
        """
        Parameters
        ----------
        estimator: Estimator
            The estimator to optimize

        param_grid: dict
            Dictionary containing params to optimize
        """
        self.estimator = estimator
        self.param_grid = param_grid

    @staticmethod
    def _train_estimator(estimator, metrics, data, cv, n_jobs, verbose):
        """
        Helper method to wrap training step in a logger call

        Parameters
        ----------
        estimator: Estimator
            Estimator to train

        metrics: List[str]
            List of metrics to calculate for the estimator

        data: Dataset
            Dataset to train estimator on

        cv: Any
            Either a CV object from sklearn or an int to specify number of folds

        verbose: int
            Verbosity level of the method


        n_jobs: int
            Number of parallel jobs to train

        Returns
        -------
        Result
        """
        logger.info("Training...")
        logger.debug("Training %s", estimator)

        result = Result.from_estimator(
            estimator=estimator,
            metrics=metrics,
            data=data,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        logger.info("Finished...")
        logger.debug("Finished %s", estimator)
        logger.info("Result: %s", result)
        return result

    def _train_estimators(
        self,
        estimators: Iterable[Estimator],
        metrics: List[str],
        data: Dataset,
        n_jobs: int,
        cv: Any,
        verbose: int,
    ):
        """
        Helper method to train a given set of estimators

        Parameters
        ----------
        estimators: Iterable[Estimator]
            An iterable of the estimators to train

        metrics: List[str]
            List of metrics to calculate

        data: Dataset
            Dataset to train estimator on

        n_jobs: int
            Number of parallel jobs to train

        cv: Any
            Either a CV object from sklearn or an int to specify number of folds

        verbose: int
            Verbosity level of the method

        Returns
        -------
        ResultGroup
            A list of Results
        """

        results = [
            self._train_estimator(
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
        """
        Search for the best parameters given the implemented search strategy

        Parameters
        ----------
        data: Dataset
            Input dataset to train on

        metrics: List of str
            Metrics to train on

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
        raise NotImplementedError
