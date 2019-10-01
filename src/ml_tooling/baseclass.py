import pathlib
from contextlib import contextmanager
from itertools import product
from typing import Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import joblib
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, fit_grid_point, check_cv

from ml_tooling.config import DefaultConfig, ConfigGetter
from ml_tooling.data.base_data import Dataset
from ml_tooling.logging.logger import create_logger
from ml_tooling.logging.log_estimator import log_results
from ml_tooling.result.viz import RegressionVisualize, ClassificationVisualize
from ml_tooling.result import Result, CVResult, ResultGroup
from ml_tooling.utils import (
    MLToolingError,
    _create_param_grid,
    _validate_estimator,
    DataSetError,
)

logger = create_logger("ml_tooling")


class Model:
    """
    Base class for Models
    """

    _config = None
    config = ConfigGetter()

    def __init__(self, estimator):
        self.estimator: Union[BaseEstimator, Pipeline] = _validate_estimator(estimator)
        self.result: Optional[Result] = None

        if self.is_classifier:
            self._plotter = ClassificationVisualize

        elif self.is_regressor:
            self._plotter = RegressionVisualize

    @property
    def is_classifier(self):
        return is_classifier(self.estimator)

    @property
    def is_regressor(self):
        return is_regressor(self.estimator)

    @property
    def estimator_name(self):
        class_name = self.estimator.__class__.__name__

        if class_name == "Pipeline":
            return self.estimator.steps[-1][1].__class__.__name__

        return class_name

    @classmethod
    def load_estimator(cls, path: str) -> "Model":
        """
        Instantiates the class with a joblib pickled estimator.

        Parameters
        ----------
        path: str, optional
            Path to estimator pickle file

        Example
        -------
        Having defined ModelData, we can load a trained estimator from disk::

            my_estimator = Model.load_estimator('path/to/estimator')

        We now have a trained estimator loaded.


        Returns
        -------
        Model
            Instance of Model with a saved estimator
        """
        estimator_file = pathlib.Path(path)
        estimator = joblib.load(estimator_file)
        instance = cls(estimator)
        logger.info(f"Loaded {instance.estimator_name}")
        return instance

    def save_estimator(self, path: str) -> pathlib.Path:
        """
        Saves the estimator as a binary file.


        Parameters
        ----------
        path : str
            Path to save estimator

        Example
        -------

        If we have trained an estimator and we want to save it to disk we can write::

            estimator.save('path/to/folder/filename.pkl')

        to save in the given folder.

        Returns
        -------
        pathlib.Path
            The path to where the estimator file was saved

        """

        estimator_file = pathlib.Path(path)

        if estimator_file.is_dir():
            raise MLToolingError(
                f"Passed directory {estimator_file} - need to pass a filename"
            )
        logger.debug(f"Attempting to save estimator in {estimator_file.parent}")
        if not estimator_file.parent.exists():
            logger.debug(f"{estimator_file.parent} does not exist - creating")
            estimator_file.parent.mkdir(parents=True)

        joblib.dump(self.estimator, estimator_file)

        if self.config.LOG:
            if self.result is None:
                raise MLToolingError(
                    "You haven't scored the estimator - no results available to log"
                )

            metric_scores = {self.result.metric: float(self.result.score)}

            log_results(
                metric_scores=metric_scores,
                estimator_name=self.estimator_name,
                estimator_params=self.result.estimator_params,
                run_dir=self.config.RUN_DIR,
                estimator_path=str(estimator_file),
            )

        logger.info(f"Saved estimator to {estimator_file}")

        return estimator_file

    def make_prediction(
        self,
        data: Dataset,
        *args,
        proba: bool = False,
        use_index: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Makes a prediction given an input. For example a customer number.
        Calls `load_prediction_data(*args)` and passes resulting data to `predict()`
        on the estimator


        Parameters
        ----------
        data: Dataset
            an instantiated Dataset object

        proba: bool
            Whether prediction is returned as a probability or not.
            Note that the return value is an n-dimensional array where n = number of classes

        use_index: bool
            Whether the index from the prediction data should be used for the result.

        Returns
        -------
        pd.DataFrame
            A DataFrame with a prediction per row.
        """
        if proba is True and not hasattr(self.estimator, "predict_proba"):
            raise MLToolingError(
                f"{self.estimator_name} does not have a `predict_proba` method"
            )
        x = data.load_prediction_data(*args, **kwargs)
        try:
            if proba:
                data = self.estimator.predict_proba(x)
            else:
                data = self.estimator.predict(x)

            if use_index:
                prediction = pd.DataFrame(data=data, index=x.index)
            else:
                prediction = pd.DataFrame(data=data)

            return prediction

        except NotFittedError:
            message = (
                f"You haven't fitted the estimator. Call 'train_estimator' "
                f"or 'score_estimator' first"
            )
            raise MLToolingError(message) from None

    @property
    def default_metric(self):
        """
        Finds estimator_type for estimator in a ModelData class and returns default
        metric for this class as configured in .config. If passed estimator is a Pipeline,
        assume last step is the estimator.

        Returns
        -------
        str
            Name of the metric

        """

        return (
            self.config.CLASSIFIER_METRIC
            if self.is_classifier
            else self.config.REGRESSION_METRIC
        )

    @default_metric.setter
    def default_metric(self, metric):
        if self.is_classifier:
            self.config.CLASSIFIER_METRIC = metric
        else:
            self.config.REGRESSION_METRIC = metric

    @classmethod
    def test_estimators(
        cls,
        data: Dataset,
        estimators: Sequence,
        metric: Optional[str] = None,
        cv: Union[int, bool] = False,
        log_dir: str = None,
    ) -> Tuple["Model", ResultGroup]:
        """
        Trains each estimator passed and returns a sorted list of results

        Parameters
        ----------
        data: Dataset
            An instantiated Dataset object with train_test data

        estimators: Sequence
            List of estimators to train

        metric: str, optional
            Metric to use in scoring of estimators

        cv: int, bool
            Whether or not to use cross-validation. If an int is passed, use that many folds

        log_dir: str, optional
            Where to store logged estimators. If None, don't log

        Returns
        -------
        List of Result objects
        """
        results = []

        for i, estimator in enumerate(estimators, start=1):
            logger.info(
                f"Training estimator {i}/{len(estimators)}: " f"{cls.estimator_name}"
            )
            challenger_estimator = cls(estimator)
            result = challenger_estimator.score_estimator(
                data=data, metric=metric, cv=cv
            )
            results.append(result)
            if log_dir:
                result.log_estimator(log_dir)

        results.sort(reverse=True)
        best_estimator = results[0].estimator
        logger.info(
            f"Best estimator: {results[0].estimator_name} - "
            f"{results[0].metric}: {results[0].score}"
        )

        return cls(best_estimator), ResultGroup(results)

    def train_estimator(self, data: Dataset) -> "Model":
        """Loads all training data and trains the estimator on all data.
        Typically used as the last step when estimator tuning is complete.

        .. warning::
            This will set self.result attribute to None. This method trains the estimator
            using all the data, so there is no validation data to measure results against

        Returns
        -------
        Model
            Returns an estimator trained on all the data, with no train-test split

        """
        logger.info("Training estimator...")
        self.estimator.fit(data.x, data.y)
        # Prevent confusion, as train_estimator does not return a result
        self.result = None
        logger.info("Estimator trained!")

        return self

    def score_estimator(
        self, data: Dataset, metric: Optional[str] = None, cv: Optional[int] = False
    ) -> "Result":
        """Scores the estimator based on training data from `data` and validates based on validation
        data from `data`.

        Defaults to no cross-validation. If you want to cross-validate the results,
        pass number of folds to cv. If cross-validation is used, `score_estimator` only
        cross-validates on training data and doesn't use the validation data.

        Returns a :class:`~ml_tooling.result.result.Result` object containing all result parameters

        Parameters
        ----------
        data: Dataset
            An instantiated Dataset object with create_train_test called

        metric: string
            Metric to use for scoring the estimator. Any sklearn metric string

        cv: int, optional
            Whether or not to use cross validation. Number of folds if an int is passed
            If False, don't use cross validation

        Returns
        -------
        Result
            A Result object that contains the results of the scoring
        """
        metric = self.default_metric if metric is None else metric
        logger.info("Scoring estimator...")

        if not data.has_validation_set:
            raise DataSetError("Must run create_train_test first!")

        self.estimator.fit(data.train_x, data.train_y)

        if cv:
            logger.info("Cross-validating...")
            self.result = self._score_estimator_cv(data, self.estimator, metric, cv)

        else:
            self.result = self._score_estimator(data, self.estimator, metric)

        if self.config.LOG:
            result_file = self.result.log_estimator(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")
        return self.result

    def gridsearch(
        self,
        data: Dataset,
        param_grid: dict,
        metric: Optional[str] = None,
        cv: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> Tuple[BaseEstimator, ResultGroup]:
        """
        Runs a gridsearch on the estimator with the passed in parameter grid.
        Ensure that it works inside a pipeline as well.

        Parameters
        ----------
        data: Dataset
            An instance of a DataSet object

        param_grid: dict
            Parameters to use for grid search

        metric: str, optional
            Metric to use for scoring. Defaults to value in
            :attr:`self.default_metric`

        cv: int, optional
            Cross validation to use. Defaults to value in :attr:`config.CROSS_VALIDATION`

        n_jobs: int, optional
            How many cores to use. Defaults to value in :attr:`config.N_JOBS`.

        Returns
        -------
        best_estimator: BaseEstimator
            Best estimator as found by the gridsearch

        result_group: ResultGroup
            ResultGroup object containing each individual score
        """

        baseline_estimator = clone(self.estimator)
        metric = self.default_metric if metric is None else metric
        n_jobs = self.config.N_JOBS if n_jobs is None else n_jobs
        cv = self.config.CROSS_VALIDATION if cv is None else cv
        cv = check_cv(cv, data.train_y, is_classifier(baseline_estimator))  # Stratify?
        self.result = None  # Fixes pickling recursion error in joblib

        logger.debug(f"Cross-validating with {cv}-fold cv using {metric}")
        logger.debug(f"Gridsearching using {param_grid}")
        param_grid = list(_create_param_grid(self.estimator, param_grid))
        logger.info("Starting gridsearch...")

        parallel = joblib.Parallel(n_jobs=n_jobs, verbose=self.config.VERBOSITY)

        out = parallel(
            joblib.delayed(fit_grid_point)(
                X=data.train_x,
                y=data.train_y,
                estimator=clone(baseline_estimator),
                train=train,
                test=test,
                scorer=get_scorer(metric),
                verbose=self.config.VERBOSITY,
                parameters=parameters,
            )
            for parameters, (train, test) in product(
                param_grid, cv.split(data.train_x, data.train_y, None)
            )
        )

        scores = [
            np.array([score[0] for score in out if score[1] == par])
            for par in param_grid
        ]

        results = [
            CVResult(
                baseline_estimator.set_params(**param),
                None,
                cv.n_splits,
                scores[i],
                metric,
            )
            for i, param in enumerate(param_grid)
        ]

        logger.info("Done!")

        self.result = ResultGroup(results)

        if self.config.LOG:
            result_file = self.result.log_estimator(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")

        return results[0].estimator, self.result

    @contextmanager
    def log(self, run_name: str):
        """:meth:`log` is a context manager that lets you turn on logging for any scoring methods
        that follow. You can pass a log_dir to specify a subdirectory to store the estimator in.
        The output is a yaml file recording estimator parameters, package version numbers,
        metrics and other useful information

        .. todo::

            Include example of output yaml

        Parameters
        ----------
        run_name: str
            Name of the folder to save the details in

        Example
        --------
        If we want to log an estimator run in the `score` folder we can write::

             with estimator.log('score'):
                estimator.score_estimator

        This will save the results of `estimator.score_estimator()` to `runs/score/`

        """
        old_dir = self.config.RUN_DIR
        self.config.LOG = True
        self.config.RUN_DIR = self.config.RUN_DIR.joinpath(run_name)
        try:
            yield
        finally:
            self.config.LOG = False
            self.config.RUN_DIR = old_dir

    def _score_estimator(self, data, estimator, metric: str) -> Result:
        """
        Scores estimator with a given score function.

        Parameters
        ----------
        data: Dataset
            An instantiated Dataset object
        estimator: BaseEstimator, Pipeline
            Estimator to evaluate

        metric: string
            Which scoring function to use


        Returns
        -------
        Result object

        """

        scoring_func = get_scorer(metric)

        score = scoring_func(estimator, data.test_x, data.test_y)
        viz = self._plotter(estimator=estimator, config=self.config, data=data)

        result = Result(estimator=estimator, viz=viz, score=score, metric=metric)

        logger.info(f"{self.estimator_name} - {metric}: {score}")

        return result

    def _score_estimator_cv(self, data, estimator, metric=None, cv=None) -> CVResult:
        """
        Scores estimator with given metric using cross-validation

        Parameters
        ----------
        data: Dataset
            An instantiated DataSet object with train_test split

        estimator: BaseEstimator
            A sklearn-compatible estimator to use for scoring

        metric: string
            Which scoring function to use

        cv: int, optional
            How many folds to use - if None, use default configuration

        Returns
        -------
        CVResult
        """
        cv = self.config.CROSS_VALIDATION if cv is None else cv

        scores = cross_val_score(
            estimator,
            data.train_x,
            data.train_y,
            cv=cv,
            scoring=metric,
            n_jobs=self.config.N_JOBS,
            verbose=self.config.VERBOSITY,
        )

        viz = self._plotter(estimator=estimator, config=self.config, data=data)

        result = CVResult(
            estimator=estimator, viz=viz, metric=metric, cross_val_scores=scores, cv=cv
        )

        if self.config.LOG:
            result.log_estimator(self.config.RUN_DIR)

        logger.info(f"{self.estimator_name} - {metric}: {np.mean(scores)}")
        return result

    @classmethod
    def reset_config(cls):
        """
        Reset configuration to default
        """
        cls._config = DefaultConfig()

        return cls

    def __repr__(self):
        return f"<Model: {self.estimator_name}>"
