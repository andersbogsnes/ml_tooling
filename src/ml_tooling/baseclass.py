import abc
import pathlib
from typing import Union, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError

from .result import Result
from .utils import MLToolingError, get_git_hash, find_model_file, get_model_name
from .config import DefaultConfig
from .result import RegressionVisualize, ClassificationVisualize

Data = Union[pd.DataFrame, np.ndarray]


class BaseClassModel(metaclass=abc.ABCMeta):
    """
    Base class for Models
    """

    config = DefaultConfig()

    def __init__(self, model):
        self.model = model
        self.model_type = model._estimator_type
        self.model_name = get_model_name(model)
        self.x = None
        self.y = None
        self.result = None
        self._plotter = None

        if self.model_type == 'classifier':
            self._plotter = ClassificationVisualize
        if self.model_type == 'regressor':
            self._plotter = RegressionVisualize

    @abc.abstractmethod
    def get_training_data(self) -> Tuple[Data, Data]:
        """
        Gets training data, returning features and labels

        :return:
            features, labels
        """

    @abc.abstractmethod
    def get_prediction_data(self, *args) -> Data:
        """
        Gets data to predict a given observation

        :return:
            features
        """

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

    def _load_data(self) -> Tuple[Data, Data]:
        """
        Internal method for loading data into class

        :return:
            self.x, self.y
        """
        if self.x is None or self.y is None:
            self.x, self.y = self.get_training_data()
        return self.x, self.y

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
            if proba is True:
                return self.model.predict_proba(x)
            else:
                return self.model.predict(x)

        except NotFittedError as e:
            message = f"You haven't fitted the model. Call 'train_model' or 'score_model' first"
            raise MLToolingError(message) from None

    @classmethod
    def test_models(cls,
                    models: Sequence,
                    metric: Optional[str] = None) -> Tuple['BaseClassModel', List[Result]]:
        """
        Trains each model passed and returns a sorted list of results

        :param models:
            List of models to train

        :param metric:
            Metric to use in scoring of model

        :return:
            List of Result
        """
        results = []
        for model in models:
            challenger_model = cls(model)
            result = challenger_model.score_model(metric=metric)
            results.append(result)
        results.sort(reverse=True)
        best_model = results[0].model

        return cls(best_model), results

    def train_model(self) -> 'BaseClassModel':
        self._load_data()
        self.model.fit(self.x, self.y)
        return self

    def score_model(self, metric=None, cv=None) -> 'Result':
        """
        Loads training data and returns a Result object containing
        visualization and cross-validated scores

        :param metric:
            Metric to score model on. Any sklearn-compatible string or scoring function

        :param cv:
            Cross validator to use. Number of folds if an int is passed,
            else any sklearn compatible CV instance

        :return:
            Result
        """
        self._load_data()

        if self.model_type == 'classifier':
            metric = self.config.CLASSIFIER_METRIC if metric is None else metric
            stratify = self.y
        else:
            metric = self.config.REGRESSION_METRIC if metric is None else metric
            stratify = None

        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, stratify=stratify)
        if isinstance(train_x, pd.DataFrame) and isinstance(test_x, pd.DataFrame):
            train_x = train_x.reset_index(drop=True)
            test_x = test_x.reset_index(drop=True)

        cv = self.config.CROSS_VALIDATION if cv is None else cv
        scores = cross_val_score(self.model,
                                 train_x,
                                 train_y,
                                 cv=cv,
                                 scoring=metric,
                                 n_jobs=self.config.N_JOBS,
                                 verbose=self.config.VERBOSITY,
                                 )

        self.model.fit(train_x, train_y)

        viz = self._plotter(model=self.model,
                            config=self.config,
                            train_x=train_x,
                            train_y=train_y,
                            test_x=test_x,
                            test_y=test_y)

        self.result = Result(
            model=self.model,
            viz=viz,
            model_name=self.model_name,
            model_params=self.model.get_params(),
            metric=metric,
            cross_val_scores=scores,
            cross_val_mean=np.mean(scores),
            cross_val_std=np.std(scores),
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
