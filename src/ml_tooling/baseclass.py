import abc
import pathlib
from contextlib import contextmanager
from itertools import product
from typing import Tuple, Optional, Sequence, Union, Any

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, fit_grid_point, check_cv

from ml_tooling.data import Data
from .config import DefaultConfig, ConfigGetter
from ml_tooling.logging.logger import create_logger
from ml_tooling.logging.log_model import log_model
from ml_tooling.result.viz import RegressionVisualize, ClassificationVisualize
from .result import Result, CVResult, ResultGroup
from .utils import (
    MLToolingError,
    _get_model_name,
    get_git_hash,
    DataType,
    find_model_file,
    get_scoring_func,
    _create_param_grid,
    _validate_model
)

logger = create_logger('ml_tooling')


class BaseClassModel(metaclass=abc.ABCMeta):
    """
    Base class for Models
    """

    _config = None
    _data = None
    config = ConfigGetter()

    def __init__(self, model):
        self.model = _validate_model(model)
        self.model_name = _get_model_name(model)
        self.result = None
        self._plotter = None

        if self.model._estimator_type == 'classifier':
            self._plotter = ClassificationVisualize

        if self.model._estimator_type == 'regressor':
            self._plotter = RegressionVisualize

    @abc.abstractmethod
    def get_training_data(self) -> Tuple[DataType, DataType]:
        """
        Gets training data, returning features and labels

        Returns
        -------
        features : pd.DataFrame
            Features to use for training

        labels : pd.DataFrame
            Labels to use for training
        """

    @abc.abstractmethod
    def get_prediction_data(self, *args, **kwargs) -> DataType:
        """
        Gets data to predict a given observation

        Returns
        -------
        pd.DataFrame
            Features to use in prediction
        """

    @classmethod
    def setup_model(cls) -> 'BaseClassModel':
        """
        Setup an untrained model from scratch - create pipeline and model and load the class

        Returns
        -------
        BaseClassModel
        """
        raise NotImplementedError

    @property
    def class_name(self):
        return self.__class__.__name__

    @classmethod
    def load_model(cls, path: Optional[str] = None) -> 'BaseClassModel':
        """
        Load previously saved model from path

        Parameters
        ----------
        path: str, optional
            Where to load the model from. If None, will load newest model that includes
            the model name and class name

        Returns
        -------
        BaseClassModel
            Instance of saved model
        """
        path = cls.config.MODEL_DIR if path is None else pathlib.Path(path)
        model_file = find_model_file(path)
        model = joblib.load(model_file)
        instance = cls(model)
        logger.info(f"Loaded {instance.model_name} for {cls.__name__}")
        return instance

    @property
    def data(self):
        if self.__class__._data is None:
            self.__class__._data = self._load_data()
        return self.__class__._data

    def _load_data(self) -> Data:
        """
        Internal method for loading data into class

        Returns
        -------
        Data
            Data object containing train-test split as well as original x and y
        """

        logger.debug("No data loaded - loading...")
        x, y = self.get_training_data()

        stratify = y if self.model._estimator_type == 'classifier' else None
        logger.debug("Creating train/test...")
        return Data.with_train_test(x,
                                    y,
                                    stratify=stratify,
                                    test_size=self.config.TEST_SIZE,
                                    seed=self.config.RANDOM_STATE)

    def _generate_filename(self):
        return f"{self.__class__.__name__}_{self.model_name}_{get_git_hash()}.pkl"

    def save_model(self, path: Optional[str] = None) -> pathlib.Path:
        """
        Save model to disk. Defaults to current directory.

        Parameters
        ----------
        path: str
            Full path of where to save the model

        Returns
        -------
        self
        """
        save_name = self._generate_filename()
        current_dir = (self.config.MODEL_DIR
                       if path is None
                       else pathlib.Path(path)
                       )

        logger.debug(f"Attempting to save model in {current_dir}")
        if not current_dir.exists():
            logger.debug(f"{current_dir} does not exist - creating")
            current_dir.mkdir(parents=True)

        model_file = current_dir.joinpath(save_name)
        joblib.dump(self.model, model_file)

        if self.config.LOG:
            if self.result is None:
                raise MLToolingError("You haven't scored the model - no results available to log")

            metric_scores = {self.result.metric: float(self.result.score)}

            log_model(metric_scores=metric_scores,
                      model_name=self.model_name,
                      model_params=self.result.model_params,
                      run_dir=self.config.RUN_DIR,
                      model_path=str(model_file))

        logger.info(f"Saved model to {model_file}")
        return model_file

    def make_prediction(self,
                        input_data: Any,
                        proba: bool = False,
                        use_index: bool = False) -> pd.DataFrame:
        """
        Returns model prediction for given input data

        Parameters
        ----------
        input_data: any
            Defined in .get_prediction_data

        proba: bool
            Whether prediction is returned as a probability or not.
            Note that the return value is an n-dimensional array where n = number of classes

        use_index: bool
            Whether the row names from the  prediction data should be used for the result.

        Returns
        -------
        Prediction: pd.DataFrame
            Depending on whether or not predict_proba is set, will contain prediction
        """
        if proba is True and not hasattr(self.model, 'predict_proba'):
            raise MLToolingError(f"{self.model_name} does not have a `predict_proba` method")

        x = self.get_prediction_data(input_data)

        try:
            if proba:
                data = self.model.predict_proba(x)
            else:
                data = self.model.predict(x)

            if use_index:
                prediction = pd.DataFrame(data=data, index=x.index)
            else:
                prediction = pd.DataFrame(data=data)

            return prediction

        except NotFittedError:
            message = f"You haven't fitted the model. Call 'train_model' or 'score_model' first"
            raise MLToolingError(message) from None

    @property
    def default_metric(self):
        """
        Finds estimator_type for estimator in a BaseClassModel and returns default
        metric for this class stated in .config. If passed estimator is a Pipeline,
        assume last step is the estimator.

        Returns
        -------
        str
            Name of the metric

        """

        return (self.config.CLASSIFIER_METRIC
                if self.model._estimator_type == 'classifier'
                else self.config.REGRESSION_METRIC)

    @default_metric.setter
    def default_metric(self, metric):
        if self.model._estimator_type == 'classifier':
            self.config.CLASSIFIER_METRIC = metric
        else:
            self.config.REGRESSION_METRIC = metric

    @classmethod
    def test_models(cls,
                    models: Sequence,
                    metric: Optional[str] = None,
                    cv: Union[int, bool] = False,
                    log_dir: str = None) -> Tuple['BaseClassModel', ResultGroup]:
        """
        Trains each model passed and returns a sorted list of results

        Parameters
        ----------
        models: Listlike
            List of models to train

        metric: str, optional
            Metric to use in scoring of model

        cv: int, bool
            Whether or not to use cross-validation. If an int is passed, use that many folds

        log_dir: str, optional
            Where to store logged models. If None, don't log

        Returns
        -------
        List of Result objects
        """
        results = []

        for i, model in enumerate(models, start=1):
            logger.info(f"Training model {i}/{len(models)}: {_get_model_name(model)}")
            challenger_model = cls(model)
            result = challenger_model.score_model(metric=metric, cv=cv)
            results.append(result)
            if log_dir:
                result.log_model(log_dir)

        results.sort(reverse=True)
        best_model = results[0].model
        logger.info(
            f"Best model: {results[0].model_name} - {results[0].metric}: {results[0].score}")

        return cls(best_model), ResultGroup(results)

    def train_model(self) -> 'BaseClassModel':
        """
        Trains the model on the full dataset.
        Used to prepare for production

        Returns
        -------
        self

        """
        logger.info("Training model...")
        self.model.fit(self.data.x, self.data.y)
        self.result = None  # Prevent confusion, as train_model does not return a result
        logger.info("Model trained!")

        return self

    def score_model(self, metric: Optional[str] = None, cv: Optional[int] = False) -> 'Result':
        """
        Loads training data and returns a Result object containing
        visualization and cross-validated scores

        Parameters
        ----------
        metric: string
            Metric to use for scoring the model. Any sklearn metric string

        cv: int, optional
            Whether or not to use cross validation. Number of folds if an int is passed
            If False, don't use cross validation

        Returns
        -------
        Result object
        """
        metric = self.default_metric if metric is None else metric
        logger.info("Scoring model...")
        self.model.fit(self.data.train_x, self.data.train_y)

        if cv:  # TODO handle case of Sklearn CV class
            logger.info("Cross-validating...")
            self.result = self._score_model_cv(self.model, metric, cv)

        else:
            self.result = self._score_model(self.model, metric)

        if self.config.LOG:
            result_file = self.result.log_model(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")
        return self.result

    def gridsearch(self,
                   param_grid: dict,
                   metric: Optional[str] = None,
                   cv: Optional[int] = None,
                   n_jobs: Optional[int] = None) -> Tuple[BaseEstimator, ResultGroup]:
        """
        Grid search model with parameters in param_grid.
        Param_grid automatically adds prefix from last step if using pipeline

        Parameters
        ----------
        param_grid: dict
            Parameters to use for grid search

        metric: str, optional
            Metric to use for scoring. Defaults to r2 for regressors and accuracy for classifiers

        cv: int, optional
            Cross validation to use. Defaults to 10 based on value in config

        n_jobs: int, optional
            How many worker process too spawn. Defaults to -1  based on value
            in config (one for each physical core).

        Returns
        -------
        best_model: sklearn.estimator
            Best model as found by the gridsearch

        result_group: ResultGroup
            ResultGroup object containing each individual score
        """

        baseline_model = clone(self.model)
        train_x, train_y = self.data.train_x, self.data.train_y
        metric = self.default_metric if metric is None else metric
        n_jobs = self.config.N_JOBS if n_jobs is None else n_jobs
        cv = self.config.CROSS_VALIDATION if cv is None else cv
        cv = check_cv(cv, train_y, baseline_model._estimator_type == 'classifier')  # Stratify?
        self.result = None  # Fixes pickling recursion error in joblib

        logger.debug(f"Cross-validating with {cv}-fold cv using {metric}")
        logger.debug(f"Gridsearching using {param_grid}")
        param_grid = list(_create_param_grid(self.model, param_grid))
        logger.info("Starting gridsearch...")

        parallel = joblib.Parallel(n_jobs=n_jobs, verbose=self.config.VERBOSITY)

        out = parallel(
            joblib.delayed(fit_grid_point)(X=train_x, y=train_y,
                                           estimator=clone(baseline_model),
                                           train=train, test=test,
                                           scorer=get_scoring_func(metric),
                                           verbose=self.config.VERBOSITY,
                                           parameters=parameters) for parameters, (train, test)
            in product(param_grid, cv.split(train_x, train_y, None)))

        scores = [np.array([score[0] for score in out if score[1] == par]) for par in param_grid]

        results = [CVResult(baseline_model.set_params(**param), None, cv.n_splits, scores[i],
                            metric) for i, param in enumerate(param_grid)]

        logger.info("Done!")

        self.result = ResultGroup(results)

        if self.config.LOG:
            result_file = self.result.log_model(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")

        return results[0].model, self.result

    @contextmanager
    def log(self, run_name: str):
        """
        Log this run by saving a yaml file in the ./<RUN_DIR>/<run_name> folder

        Parameters
        ----------
        run_name: str
            Name of the folder to save the details in

        """
        old_dir = self.config.RUN_DIR
        self.config.LOG = True
        self.config.RUN_DIR = self.config.RUN_DIR.joinpath(run_name)
        try:
            yield
        finally:
            self.config.LOG = False
            self.config.RUN_DIR = old_dir

    def _score_model(self, model, metric: str) -> Result:
        """
        Scores model with a given score function.

        Parameters
        ----------
        model: sklearn.estimator
            Estimator to evaluate

        metric: string
            Which scoring function to use


        Returns
        -------
        Result object

        """

        scoring_func = get_scoring_func(metric)

        score = scoring_func(model, self.data.test_x, self.data.test_y)
        viz = self._plotter(model=model,
                            config=self.config,
                            data=self.data)

        result = Result(model=model,
                        viz=viz,
                        score=score,
                        metric=metric)

        logger.info(f"{_get_model_name(model)} - {metric}: {score}")

        return result

    def _score_model_cv(self,
                        model,
                        metric=None,
                        cv=None) -> CVResult:
        """
        Scores model with given metric using cross-validation

        Parameters
        ----------

        metric: string
            Which scoring function to use

        cv: int, optional
            How many folds to use - if None, use default configuration

        Returns
        -------
        CVResult
        """
        cv = self.config.CROSS_VALIDATION if cv is None else cv

        scores = cross_val_score(model,
                                 self.data.train_x,
                                 self.data.train_y,
                                 cv=cv,
                                 scoring=metric,
                                 n_jobs=self.config.N_JOBS,
                                 verbose=self.config.VERBOSITY,
                                 )

        viz = self._plotter(model=model,
                            config=self.config,
                            data=self.data)

        result = CVResult(
            model=model,
            viz=viz,
            metric=metric,
            cross_val_scores=scores,
            cv=cv
        )

        if self.config.LOG:
            result.log_model(self.config.RUN_DIR)

        logger.info(f"{_get_model_name(model)} - {metric}: {np.mean(scores)}")
        return result

    @classmethod
    def reset_config(cls):
        """
        Reset configuration to default
        """
        cls._config = DefaultConfig()

        return cls

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.model_name}>"
