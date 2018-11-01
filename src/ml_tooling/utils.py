from typing import Union

import numpy as np
import pandas as pd
from git import Repo, InvalidGitRepositoryError
import pathlib
from collections import namedtuple
from sklearn.model_selection import train_test_split

DataType = Union[pd.DataFrame, np.ndarray]


class Data:
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
                          ):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x,
                                                                                self.y,
                                                                                stratify=stratify,
                                                                                shuffle=shuffle,
                                                                                test_size=test_size)
        return self

    @classmethod
    def with_train_test(cls, x: DataType, y: DataType, stratify=None, shuffle=True, test_size=0.25):
        instance = cls(x, y)
        return instance.create_train_test(stratify=stratify, shuffle=shuffle, test_size=test_size)


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


def create_train_test(x, y,
                      stratify=None,
                      shuffle=True,
                      test_size=0.25,
                      random_state=None
                      ):
    train_x, test_x, train_y, test_y = train_test_split(x,
                                                        y,
                                                        stratify=stratify,
                                                        shuffle=shuffle,
                                                        test_size=test_size,
                                                        random_state=random_state)
    return Train_Test(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)


Data = Union[pd.DataFrame, np.ndarray]
