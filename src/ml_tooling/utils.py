import pathlib
from typing import Union, Callable

import numpy as np
import pandas as pd
from git import Repo, InvalidGitRepositoryError
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.utils import indexable

DataType = Union[pd.DataFrame, np.ndarray]


class Data:
    """
    Container for storing data. Contains both x and y, while also handling train_test_split
    """

    def __init__(self, x: DataType, y: DataType):
        self.x, self.y = indexable(x, y)
        self.train_y = None
        self.train_x = None
        self.test_y = None
        self.test_x = None

    def create_train_test(self,
                          stratify=None,
                          shuffle=True,
                          test_size=0.25,
                          ) -> 'Data':
        """
        Creates a training and testing dataset and storing it on the data object.
        :param stratify:
            What to stratify the split on. Usually y if given a classification problem
        :param shuffle:
            Whether or not to shuffle the data
        :param test_size:
            What percentage of the data will be part of the test set
        :return:
            self
        """
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x,
                                                                                self.y,
                                                                                stratify=stratify,
                                                                                shuffle=shuffle,
                                                                                test_size=test_size)
        return self

    @classmethod
    def with_train_test(cls,
                        x: DataType,
                        y: DataType,
                        stratify=None,
                        shuffle=True,
                        test_size=0.25) -> 'Data':
        """
        Creates a new instance of Data with train and test already instantiated
        :param x:
            Features
        :param y:
            Target
        :param stratify:
             What to stratify the split on. Usually y if given a classification problem
        :param shuffle:
            Whether or not to shuffle the data
        :param test_size:
            What percentage of the data will be part of the test set
        :return:
            self
        """
        instance = cls(x, y)
        return instance.create_train_test(stratify=stratify, shuffle=shuffle, test_size=test_size)


class MLToolingError(Exception):
    """Error which occurs when using the library"""
    pass


class TransformerError(Exception):
    """Error which occurs during a transform"""
    pass


def get_scoring_func(metric: str) -> Callable[[BaseEstimator,
                                               DataType,
                                               DataType], Union[int, float]]:
    """
    Looks up a scikit-learn scoring function using scikit-learns built-in sklearn.metrics.get_scorer
    :param metric:
        string name of metric to use
    :return:
        callable score function
    """
    try:
        return get_scorer(metric)
    except ValueError:
        raise MLToolingError(f"Invalid metric {metric}")


def get_git_hash() -> str:
    """
    Returns the git hash of HEAD
    :return:
        git hash of HEAD
    """
    try:
        repo = Repo(search_parent_directories=True)
    except InvalidGitRepositoryError:
        return ''
    return repo.head.object.hexsha


def find_model_file(path: str) -> pathlib.Path:
    """
    Helper to find a model file in a given directory.
    If path is a directory - returns newest model that matches the git hash
    :param path: dir or path to model
    :return:
    """
    path = pathlib.Path(path)

    if path.is_file():
        return path

    git_hash = get_git_hash()
    all_models = list(path.glob(f'*_{git_hash}.pkl'))

    if not all_models:
        raise MLToolingError(f"No models found - check your directory: {path}")

    newest_match = max(all_models, key=lambda x: x.stat().st_mtime)
    return newest_match


def _get_model_name(clf) -> str:
    """
    Returns model name based on class name. If passed classifier is a Pipeline,
    assume last step is the estimator and return that classes name
    :param clf: sklearn-compatible estimator
    :return:
    """
    if clf.__class__.__name__ == 'Pipeline':
        return clf.steps[-1][1].__class__.__name__

    return clf.__class__.__name__


def listify(collection) -> list:
    """
    Takes a given collection and returns a list of the elements
    :param collection:
        Any collection or str
    :return:
        list
    """
    if isinstance(collection, str):
        collection = [collection]

    if isinstance(collection, (tuple, set)):
        collection = list(collection)

    return collection


def _is_percent(number: Union[float, int]) -> bool:
    """
    Checks if a value is a valid percent
    :param number:
        The number to validate
    :return:
        bool
    """
    if isinstance(number, float):
        if number > 1 or number < 0:
            raise ValueError(f"Floats only valid between 0 and 1. Got {number}")
        return True
    return False


def _most_freq(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mode of X
    :param X:
        DataFrame to calculate mode over
    :return:
        DataFrame of modes
    """
    return pd.DataFrame.mode(X).iloc[0]


def _create_param_grid(pipe: Pipeline, param_grid: dict) -> ParameterGrid:
    """
    Creates a parameter grid from a pipeline
    :param pipe:
        Sklearn Pipeline
    :param param_grid:
        dict of parameters to search over
    :return:
        sklearn ParameterGrid
    """
    if not isinstance(pipe, Pipeline):
        return ParameterGrid(param_grid)

    step_name = pipe.steps[-1][0]

    step_dict = {f"{step_name}__{param}" if step_name not in param else param: value
                 for param, value
                 in param_grid.items()}

    return ParameterGrid(step_dict)


def _validate_model(model):
    """
    Ensures that model runs
    :param model:
    :return:
    """
    if hasattr(model, '_estimator_type'):
        return model

    if isinstance(model, Pipeline):
        raise MLToolingError("You passed a Pipeline without an estimator as the last step")

    raise MLToolingError(f"Expected a Pipeline or Estimator - got {type(model)}")
