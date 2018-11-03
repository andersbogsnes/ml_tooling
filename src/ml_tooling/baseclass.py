import abc
import pathlib
from typing import List, Tuple, Optional, Sequence, Union

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from .result import Result, CVResult
from .utils import (
    MLToolingError,
    get_model_name,
    get_git_hash,
    DataType,
    find_model_file,
    Data,
    get_scoring_func,
)
from .config import DefaultConfig
from .result import RegressionVisualize, ClassificationVisualize


class BaseClassModel(metaclass=abc.ABCMeta):
    """
    Base class for Models
    """

    config = DefaultConfig()

    def __init__(self, model):
        self.model = model
        self.model_type = model._estimator_type
        self.model_name = get_model_name(model)
        self.data = None
        self.result = None
        self._plotter = None

        if self.model_type == 'classifier':
            self._plotter = ClassificationVisualize
        if self.model_type == 'regressor':
            self._plotter = RegressionVisualize

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

    @classmethod
    def load_model(cls, path) -> 'BaseClassModel':
        """
        Load previously saved model from path

        :return:
            cls
        """

        model_file = find_model_file(path)
        model = joblib.load(model_file)
        return cls(model)

    def _load_data(self) -> Data:
        """
        Internal method for loading data into class

        :return:
            self.x, self.y
        """
        if self.data is None:
            x, y = self.get_training_data()
            self.data = Data(x, y)
        return self.data

    def _generate_filename(self):
        return f"{self.__class__.__name__}_{self.model_name}_{get_git_hash()}.pkl"

    def save_model(self, path=None) -> 'BaseClassModel':
        """
        Save model to disk. Defaults to current directory.

        :param path:
            Full path to save the model

        :return:
            self
        """
        save_name = self._generate_filename()
        current_dir = pathlib.Path.cwd() if path is None else pathlib.Path(path)

        if not current_dir.exists():
            current_dir.mkdir(parents=True)

        model_file = current_dir.joinpath(save_name)
        joblib.dump(self.model, model_file)
        return self

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
                    cv: Union[int, bool] = False) -> Tuple['BaseClassModel', List[Result]]:
        """
        Trains each model passed and returns a sorted list of results

        :param models:
            List of models to train

        :param metric:
            Metric to use in scoring of model

        :param cv:
            Whether or not to use cross-validation. If an int is passed, use that many folds

        :return:
            List of Result
        """
        results = []
        for model in models:
            challenger_model = cls(model)
            result = challenger_model.score_model(metric=metric, cv=cv)
            results.append(result)
        results.sort(reverse=True)
        best_model = results[0].model

        return cls(best_model), results

    def train_model(self) -> 'BaseClassModel':
        """
        Trains the model on the full dataset.
        Used to prepare for production
        :return:
            self
        """
        self._load_data()
        self.model.fit(self.data.x, self.data.y)
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
        self._load_data()

        if self.model_type == 'classifier':
            metric = self.config.CLASSIFIER_METRIC if metric is None else metric
            stratify = self.data.y

        else:
            metric = self.config.REGRESSION_METRIC if metric is None else metric
            stratify = None

        self.data.create_train_test(stratify=stratify)
        self.model.fit(self.data.train_x, self.data.train_y)

        if cv:  # TODO handle case of Sklearn CV class
            return self._score_model_cv(metric, cv)

        return self._score_model(metric)

    def _score_model(self, metric: str) -> Result:
        """
        Scores model with a given score function.
        :param metric:
            string of which scoring function to use
        :return:
        """
        # TODO support any given sklearn scorer - must check that it is a scorer
        scoring_func = get_scoring_func(metric)

        score = scoring_func(self.model, self.data.test_x, self.data.test_y)
        viz = self._plotter(model=self.model,
                            config=self.config,
                            data=self.data)

        self.result = Result(model=self.model,
                             model_name=self.model_name,
                             viz=viz,
                             model_params=self.model.get_params(),
                             score=score,
                             metric=metric)

        return self.result

    def _score_model_cv(self, metric: Optional[str] = None, cv: Optional[int] = None) -> CVResult:
        """
        Scores model with given metric using cross-validation
        :param metric:
            string of which scoring function to use
        :param cv:
            How many folds to use - if None, use default configuration
        :return:
        """
        cv = self.config.CROSS_VALIDATION if cv is None else cv

        scores = cross_val_score(self.model,
                                 self.data.train_x,
                                 self.data.train_y,
                                 cv=cv,
                                 scoring=metric,
                                 n_jobs=self.config.N_JOBS,
                                 verbose=self.config.VERBOSITY,
                                 )

        viz = self._plotter(model=self.model,
                            config=self.config,
                            data=self.data)

        self.result = CVResult(
            model=self.model,
            viz=viz,
            model_name=self.model_name,
            model_params=self.model.get_params(),
            metric=metric,
            cross_val_scores=scores,
            cross_val_mean=np.mean(scores),
            cross_val_std=np.std(scores),
            cv=cv
        )

        return self.result

    @classmethod
    def reset_config(cls):
        """
        Reset configuration to default
        """
        cls.config = DefaultConfig()
        return cls

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.model_name}>"
