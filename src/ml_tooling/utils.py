from typing import Union, Callable

import numpy as np
import pandas as pd
from git import Repo, InvalidGitRepositoryError
import pathlib

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics.scorer import (explained_variance_scorer,
                                    r2_scorer,
                                    fowlkes_mallows_scorer,
                                    neg_median_absolute_error_scorer,
                                    neg_mean_absolute_error_scorer,
                                    neg_mean_squared_error_scorer,
                                    neg_mean_squared_log_error_scorer,
                                    accuracy_scorer,
                                    roc_auc_scorer,
                                    balanced_accuracy_scorer,
                                    average_precision_scorer,
                                    neg_log_loss_scorer,
                                    brier_score_loss_scorer,
                                    adjusted_rand_scorer,
                                    homogeneity_scorer,
                                    completeness_scorer,
                                    v_measure_scorer,
                                    mutual_info_scorer,
                                    adjusted_mutual_info_scorer,
                                    normalized_mutual_info_scorer,
                                    )
from sklearn.model_selection import train_test_split, ParameterGrid

DataType = Union[pd.DataFrame, np.ndarray]

SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               accuracy=accuracy_scorer,
               roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)


class Data:
    """
    Container for storing data. Contains both x and y, while also handling train_test_split
    """
    def __init__(self, x: DataType, y: DataType):
        self.x = x
        self.y = y
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
    Looks up a string scoring function in the scorers dictionary
    :param metric:
        string name of metric to use
    :return:
        callable score function
    """
    try:
        return SCORERS[metric]
    except KeyError:
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


def get_model_name(clf) -> str:
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


def _get_labels(model: BaseEstimator, data: pd.DataFrame) -> np.array:
    """
    If data is a DataFrame, use columns attribute - else use [0...n] np.array
    :return:
        list-like of labels
    """
    if hasattr(data, 'columns'):
        if isinstance(model, Pipeline):
            labels = _get_labels_from_pipeline(model, data)
        else:
            labels = data.columns
    else:
        labels = np.arange(data.shape[1])

    return np.array(labels)


def _get_labels_from_pipeline(pipe: Pipeline, df: pd.DataFrame) -> np.array:
    """
    Transforms df using the transformer steps of the pipeline and then getting the column labels
    :param pipe:
        sklearn Pipeline
    :param df:
    :return:
    """
    transformers = pipe.steps[:-1]
    transform_pipe = Pipeline(transformers)
    return np.array(transform_pipe.transform(df).columns)
