import abc
from functools import total_ordering
import pathlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals import joblib
import numpy as np
from ..config import default_config
from ..visualizations import ClassificationVisualize, RegressionVisualize


@total_ordering
class Result:
    """
    Data class for holding results of model testing.
    Also implements comparison operators for finding max mean score
    """

    def __init__(self,
                 model,
                 model_name,
                 viz,
                 model_params=None,
                 cross_val_scores=None,
                 cross_val_mean=None,
                 cross_val_std=None,
                 metric=None
                 ):
        self.model = model
        self.model_name = model_name
        self.cross_val_scores = cross_val_scores
        self.cross_val_mean = cross_val_mean
        self.cross_val_std = cross_val_std
        self.model_params = model_params
        self.metric = metric
        self.plot = viz

    def __eq__(self, other):
        return self.cross_val_mean == other.cross_val_mean

    def __lt__(self, other):
        return self.cross_val_mean < other.cross_val_mean

    def __repr__(self):
        return f"<Result {self.model_name}: " \
               f"Cross-validated {self.metric}: {np.round(self.cross_val_mean, 2)} " \
               f"± {np.round(self.cross_val_std, 2)}>"


class BaseClassModel(metaclass=abc.ABCMeta):
    """
    Base class for Models
    """

    def __init__(self, model):
        self.model = model
        self.model_type = model._estimator_type
        self.model_name = model.__class__.__name__
        self.config = default_config
        self.x = None
        self.y = None
        self.result = None
        self._plotter = None

        if self.model_type == 'classifier':
            self._plotter = ClassificationVisualize
        if self.model_type == 'regressor':
            self._plotter = RegressionVisualize

    @abc.abstractmethod
    def get_training_data(self) -> tuple:
        """
        Gets training data, returning features and labels
        :return: features, labels
        """

    @abc.abstractmethod
    def get_prediction_data(self, *args):
        """Gets data to predict a given observation"""

    @classmethod
    def load_model(cls, path):
        model = joblib.load(path)
        return cls(model)

    def set_config(self, config_dict):
        """
        Update configuration using a dictionary of values
        :param config_dict: dict of config values
        :return: None
        """
        self.config.update(config_dict)
        return self

    def _load_data(self):
        if self.x is None or self.y is None:
            self.x, self.y = self.get_training_data()
        return self.x, self.y

    def save_model(self, path=None):
        current_dir = pathlib.Path.cwd().joinpath(self.model_name) if path is None else path
        joblib.dump(self.model, current_dir)
        return self

    def make_prediction(self, input_data):
        x = self.get_prediction_data(input_data)
        return self.model.predict(x)

    def train_model(self):
        self._load_data()
        self.model.fit(self.x, self.y)
        return self

    def test_model(self, metric=None):
        self._load_data()

        if self.model_type == 'classifier':
            metric = self.config['CLASSIFIER_METRIC'] if metric is None else metric
            stratify = self.y
        else:
            metric = self.config['REGRESSION_METRIC'] if metric is None else metric
            stratify = None

        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, stratify=stratify)

        scores = cross_val_score(self.model,
                                 train_x,
                                 train_y,
                                 cv=self.config['CROSS_VALIDATION'],
                                 scoring=metric,
                                 n_jobs=self.config['N_JOBS'],
                                 verbose=self.config['VERBOSITY'])

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

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.model_name}>"
