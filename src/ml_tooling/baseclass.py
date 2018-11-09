import abc
import pathlib
from contextlib import contextmanager
from typing import Tuple, Optional, Sequence, Union

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

from ml_tooling.logging import create_logger, log_model
from ml_tooling.utils import _validate_model
from .config import DefaultConfig
from .result import RegressionVisualize, ClassificationVisualize
from .result import Result, CVResult, ResultGroup
from .utils import (
    MLToolingError,
    _get_model_name,
    get_git_hash,
    DataType,
    find_model_file,
    Data,
    get_scoring_func,
    _create_param_grid
)

logger = create_logger('ml_tooling')


class BaseClassModel(metaclass=abc.ABCMeta):
    """
    Base class for Models
    """

    config = DefaultConfig()

    def __init__(self, model):
        self.model = _validate_model(model)
        self.model_name = _get_model_name(model)
        self.data = None
        self.result = None
        self._plotter = None
        self.default_metric = None

        if self.model._estimator_type == 'classifier':
            self._plotter = ClassificationVisualize
            self.default_metric = self.config.CLASSIFIER_METRIC

        if self.model._estimator_type == 'regressor':
            self._plotter = RegressionVisualize
            self.default_metric = self.config.REGRESSION_METRIC

    @abc.abstractmethod
    def get_training_data(self) -> Tuple[DataType, DataType]:
        """
        Gets training data, returning features and labels

        :return:
            features, labels
        """

    @abc.abstractmethod
    def get_prediction_data(self, *args) -> DataType:
        """
        Gets data to predict a given observation

        :return:
            features
        """

    @classmethod
    def setup_model(cls):
        """
        Setup an untrained model from scratch - create pipeline and model and load the class
        :return:
        """
        raise NotImplementedError

    @property
    def class_name(self):
        return self.__class__.__name__

    @classmethod
    def load_model(cls, path=None) -> 'BaseClassModel':
        """
        Load previously saved model from path

        :return:
            cls
        """
        path = cls.config.MODEL_DIR if path is None else pathlib.Path(path)
        model_file = find_model_file(path)
        model = joblib.load(model_file)
        instance = cls(model)
        logger.info(f"Loaded {instance.model_name} for {cls.__name__}")
        return instance

    def _load_data(self, train_test=False) -> Data:
        """
        Internal method for loading data into class

        :return:
            self.x, self.y
        """
        if self.data is None:
            logger.debug("No data loaded - loading...")
            x, y = self.get_training_data()
            if train_test:

                stratify = y if self.model._estimator_type == 'classifier' else None
                logger.debug("Creating train/test...")
                self.data = Data.with_train_test(x, y,
                                                 stratify=stratify,
                                                 test_size=self.config.TEST_SIZE)
            else:
                logger.debug("Data already loaded")
                self.data = Data(x, y)
        return self.data

    def _generate_filename(self):
        return f"{self.__class__.__name__}_{self.model_name}_{get_git_hash()}.pkl"

    def save_model(self, path=None) -> pathlib.Path:
        """
        Save model to disk. Defaults to current directory.

        :param path:
            Full path to save the model

        :return:
            self
        """
        save_name = self._generate_filename()
        current_dir = (self.config.MODEL_DIR
                       if path is None
                       else pathlib.Path(path)
                       )

        logger.debug(f"Attempting to save model in {current_dir}")
        if not current_dir.exists():
            logger.debug(f"{current_dir} doesn't exist - creating")
            current_dir.mkdir(parents=True)

        model_file = current_dir.joinpath(save_name)
        joblib.dump(self.model, model_file)

        metric_scores = {self.result.metric: float(self.result.score)}

        log_model(metric_scores=metric_scores,
                  model_name=self.model_name,
                  model_params=self.result.model_params,
                  run_dir=self.config.RUN_DIR,
                  model_path=str(model_file))

        logger.info(f"Saved model to {model_file}")
        return model_file

    def make_prediction(self, input_data, proba=False) -> np.ndarray:
        """
        Returns model prediction for given input data

        :param input_data:
            Defined in .get_prediction_data

        :param proba:
            Whether prediction is returned as a probability or not.
            Note that the return value is an n-dimensional array where n = number of classes

        :return:
            Class prediction
        """
        if proba is True and not hasattr(self.model, 'predict_proba'):
            raise MLToolingError(f"{self.model_name} doesn't have a `predict_proba` method")

        x = self.get_prediction_data(input_data)

        try:
            if proba:
                return self.model.predict_proba(x)

            return self.model.predict(x)

        except NotFittedError:
            message = f"You haven't fitted the model. Call 'train_model' or 'score_model' first"
            raise MLToolingError(message) from None

    @classmethod
    def test_models(cls,
                    models: Sequence,
                    metric: Optional[str] = None,
                    cv: Union[int, bool] = False,
                    log_dir: str = None) -> Tuple['BaseClassModel', ResultGroup]:
        """
        Trains each model passed and returns a sorted list of results

        :param models:
            List of models to train

        :param metric:
            Metric to use in scoring of model

        :param cv:
            Whether or not to use cross-validation. If an int is passed, use that many folds

        :param log_dir:
            Where to store logged models. If None, don't log

        :return:
            List of Result
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
        :return:
            self
        """
        logger.info("Training model...")
        self._load_data(train_test=False)
        self.model.fit(self.data.x, self.data.y)
        logger.info("Model trained!")

        return self

    def score_model(self, metric=None, cv=False) -> 'Result':
        """
        Loads training data and returns a Result object containing
        visualization and cross-validated scores

        :param metric:
            Metric to score model on. Any sklearn-compatible string or scoring function

        :param cv:
            Whether or not to use cross validation. Number of folds if an int is passed
            If False, don't use cross validation

        :return:
            Result
        """
        self._load_data(train_test=True)
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
                   cv: Optional[int] = None) -> Tuple[BaseEstimator, ResultGroup]:
        """
        Grid search model with parameters in param_grid.
        Param_grid automatically adds prefix from last step if using pipeline
        :param param_grid:
            Parameters to use for grid search
        :param metric:
            metric to use for scoring
        :param cv:
            Cross validation to use. Defaults to 10 based on value in config
        :return:
        """
        self._load_data(train_test=True)

        metric = self.default_metric if metric is None else metric
        cv = self.config.CROSS_VALIDATION if cv is None else cv
        logger.debug(f"Cross-validating with {cv}-fold cv using {metric}")
        logger.debug(f"Gridsearching using {param_grid}")
        param_grid = _create_param_grid(self.model, param_grid)

        baseline_model = clone(self.model)
        logger.info("Starting gridsearch...")
        parallel = joblib.Parallel(n_jobs=self.config.N_JOBS, verbose=self.config.VERBOSITY)
        results = parallel(
            joblib.delayed(self._score_model_cv)(clone(baseline_model).set_params(**param),
                                                 metric=metric,
                                                 cv=cv) for param in param_grid)
        logger.info("Done!")

        self.result = ResultGroup(results)

        if self.config.LOG:
            result_file = self.result.log_model(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")

        return results[0].model, self.result

    @contextmanager
    def log(self, run_name):
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
        :param metric:
            string of which scoring function to use
        :return:
            Result object
        """
        # TODO support any given sklearn scorer - must check that it is a scorer
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
        :param metric:
            string of which scoring function to use
        :param cv:
            How many folds to use - if None, use default configuration
        :return:
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
        cls.config = DefaultConfig()
        return cls

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.model_name}>"
