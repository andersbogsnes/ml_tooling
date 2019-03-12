import pathlib
import warnings
from typing import Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

DataType = Union[pd.DataFrame, np.ndarray]


class MLToolingError(Exception):
    """Error which occurs when using the library"""


class TransformerError(Exception):
    """Error which occurs during a transform"""


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
        from git import Repo, InvalidGitRepositoryError
    except ImportError:
        warnings.warn("Git is not installed on this system")
        return ''

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


