import pathlib
from contextlib import contextmanager
from typing import Tuple, Optional, Sequence, Union, List

import pandas as pd
import yaml
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import joblib
from sklearn.model_selection import check_cv

from ml_tooling.config import DefaultConfig, ConfigGetter
from ml_tooling.data.base_data import Dataset
from ml_tooling.logging.log_estimator import Log
from ml_tooling.logging.logger import create_logger
from ml_tooling.result import Result, ResultGroup
from ml_tooling.metrics import Metrics
from ml_tooling.search.gridsearch import prepare_gridsearch_estimators
from ml_tooling.utils import (
    MLToolingError,
    _validate_estimator,
    DataSetError,
    setup_pipeline_step,
    Estimator,
    is_pipeline,
    serialize_pipeline,
)

logger = create_logger("ml_tooling")


class Model:
    """
    Wrapper class for Estimators
    """

    _config = None
    config = ConfigGetter()

    def __init__(self, estimator):
        self.estimator: Estimator = _validate_estimator(estimator)
        self.result: Optional[Result] = None

    @property
    def is_classifier(self):
        return is_classifier(self.estimator)

    @property
    def is_regressor(self):
        return is_regressor(self.estimator)

    @property
    def is_pipeline(self):
        return is_pipeline(self.estimator)

    @property
    def estimator_name(self):
        if self.is_pipeline:
            return self.estimator.steps[-1][1].__class__.__name__

        return self.estimator.__class__.__name__

    @property
    def default_metric(self):
        """
        Defines default metric based on whether or not the estimator is a regressor or classifier.
        Then :attr:`~ml_tooling.config.DefaultConfig.CLASSIFIER_METRIC` or
        :attr:`~ml_tooling.config.DefaultConfig.CLASSIFIER_METRIC` is returned.

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

    def save_estimator(self, path: Union[str, pathlib.Path]) -> pathlib.Path:
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

            log = Log.from_result(self.result, estimator_path=estimator_file)
            log.save_log(self.config.RUN_DIR)

        logger.info(f"Saved estimator to {estimator_file}")

        return estimator_file

    def to_dict(self):
        if self.is_pipeline:
            return serialize_pipeline(self.estimator)

        return [
            {
                "module": self.estimator.__class__.__module__,
                "classname": self.estimator.__class__.__name__,
                "params": self.estimator.get_params(),
            }
        ]

    @classmethod
    def from_yaml(cls, log_file: pathlib.Path):
        log_file = pathlib.Path(log_file)
        with log_file.open("r") as f:
            log = yaml.safe_load(f)
            estimator_definition = log["estimator"]

        steps = [setup_pipeline_step(definition) for definition in estimator_definition]

        if len(steps) == 1:
            return cls(steps[0])

        return cls(Pipeline(steps))

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

    @classmethod
    def test_estimators(
        cls,
        data: Dataset,
        estimators: Sequence,
        metrics: Union[str, List[str]] = "default",
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

        metrics: str, list of str
            Metric or list of metrics to use in scoring of estimators

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
                data=data, metrics=metrics, cv=cv
            )
            results.append(result)

        results = ResultGroup(results).sort()

        if log_dir:
            results.log_estimator(pathlib.Path(log_dir))

        best_estimator = results[0].model
        logger.info(
            f"Best estimator: {best_estimator.estimator_name} - "
            f"{results[0].metrics.metric}: {results[0].metrics.score}"
        )

        return best_estimator, results

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
        self,
        data: Dataset,
        metrics: Union[str, List[str]] = "default",
        cv: Optional[int] = False,
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

        metrics: string, list of strings
            Metric or metrics to use for scoring the estimator. Any sklearn metric string

        cv: int, optional
            Whether or not to use cross validation. Number of folds if an int is passed
            If False, don't use cross validation

        Returns
        -------
        Result
            A Result object that contains the results of the scoring
        """

        if isinstance(metrics, str):
            metrics = [self.default_metric] if metrics == "default" else [metrics]

        metrics = Metrics.from_list(metrics)

        logger.info("Scoring estimator...")

        if not data.has_validation_set:
            raise DataSetError("Must run create_train_test first!")

        self.estimator.fit(data.train_x, data.train_y)

        if cv:
            logger.info("Cross-validating...")
            self.result = Result.from_model(
                model=self,
                data=data,
                metrics=metrics,
                cv=cv,
                n_jobs=self.config.N_JOBS,
                verbose=self.config.VERBOSITY,
            )

        else:
            self.result = Result.from_model(model=self, data=data, metrics=metrics)

        if self.config.LOG:
            log = Log.from_result(self.result)
            result_file = log.save_log(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")
        return self.result

    def gridsearch(
        self,
        data: Dataset,
        param_grid: dict,
        metrics: Union[str, List[str]] = "default",
        cv: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> Tuple["Model", ResultGroup]:
        """
        Runs a gridsearch on the estimator with the passed in parameter grid.
        Ensure that it works inside a pipeline as well.

        Parameters
        ----------
        data: Dataset
            An instance of a DataSet object

        param_grid: dict
            Parameters to use for grid search

        metrics: str, list of str
            Metric to use for scoring. A value of "default" sets metric equal to
            :attr:`self.default_metric`

        cv: int, optional
            Cross validation to use. Defaults to value in :attr:`config.CROSS_VALIDATION`

        n_jobs: int, optional
            How many cores to use. Defaults to value in :attr:`config.N_JOBS`.

        Returns
        -------
        best_estimator: Model
            Best estimator as found by the gridsearch

        result_group: ResultGroup
            ResultGroup object containing each individual score
        """

        if isinstance(metrics, str):
            metrics = [self.default_metric] if metrics == "default" else [metrics]

        metrics = Metrics.from_list(metrics)
        n_jobs = self.config.N_JOBS if n_jobs is None else n_jobs
        cv = self.config.CROSS_VALIDATION if cv is None else cv
        cv = check_cv(cv, data.train_y, self.is_classifier)

        logger.debug(f"Cross-validating with {cv}-fold cv using {metrics}")
        logger.debug(f"Gridsearching using {param_grid}")
        logger.info("Starting gridsearch...")

        self.result = ResultGroup(
            [
                Result.from_model(
                    model=Model(estimator),
                    data=data,
                    metrics=metrics,
                    cv=cv,
                    n_jobs=n_jobs,
                    verbose=self.config.VERBOSITY,
                )
                for estimator in prepare_gridsearch_estimators(
                    estimator=self.estimator, params=param_grid
                )
            ]
        )

        logger.info("Done!")

        if self.config.LOG:
            result_file = self.result.log_estimator(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")

        return self.result[0].model, self.result

    @contextmanager
    def log(self, run_directory: str):
        """:meth:`log` is a context manager that lets you turn on logging for any scoring methods
        that follow. You can pass a log_dir to specify a subdirectory to store the estimator in.
        The output is a yaml file recording estimator parameters, package version numbers,
        metrics and other useful information

        Parameters
        ----------
        run_directory: str
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
        self.config.RUN_DIR = self.config.RUN_DIR.joinpath(run_directory)
        try:
            yield
        finally:
            self.config.LOG = False
            self.config.RUN_DIR = old_dir

    @classmethod
    def reset_config(cls):
        """
        Reset configuration to default
        """
        cls._config = DefaultConfig()

        return cls

    def __repr__(self):
        return f"<Model: {self.estimator_name}>"
