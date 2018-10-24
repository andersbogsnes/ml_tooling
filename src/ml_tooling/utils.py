from typing import Union

import numpy as np
import pandas as pd
from git import Repo, InvalidGitRepositoryError
import pathlib


class MLToolingError(Exception):
    """Error which occurs when using the library"""
    pass


class TransformerError(Exception):
    """Error which occurs during a transform"""
    pass


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

    if isinstance(collection, tuple) or isinstance(collection, set):
        collection = list(collection)

    return collection


def _is_percent(number):
    if isinstance(number, float):
        if number > 1 or number < 0:
            raise ValueError(f"Floats only valid between 0 and 1. Got {number}")
        return True
    return False


Data = Union[pd.DataFrame, np.ndarray]
