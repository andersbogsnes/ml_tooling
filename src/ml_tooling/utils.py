import pathlib
import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

DataType = Union[pd.DataFrame, np.ndarray]


class MLToolingError(Exception):
    """Error which occurs when using the library"""


class TransformerError(MLToolingError):
    """Error which occurs during a transform"""


class DataSetError(MLToolingError):
    """Error which occurs when using a DataSet"""


def get_git_hash() -> str:
    """
    Returns the git hash of HEAD

    Returns
    -------
    str
        git hash value of HEAD
    """
    try:
        from git import Repo, InvalidGitRepositoryError
    except ImportError:
        warnings.warn("Git is not installed on this system")
        return ""

    try:
        repo = Repo(search_parent_directories=True)
    except InvalidGitRepositoryError:
        return ""
    return repo.head.object.hexsha


def find_estimator_file(path: str) -> pathlib.Path:
    """
    Helper to find a estimator file in a given directory.
    If path is a directory - returns newest estimator that matches the git hash

    Parameters
    ----------
    path: str
        directory or path to estimator. If a directory is passed, will load the newest model found

    Returns
    -------
    pathlib.Path
        path to pickled estimator
    """
    path = pathlib.Path(path)

    if path.is_file():
        return path

    all_models = list(path.glob(f"*.pkl"))

    if not all_models:
        raise MLToolingError(f"No models found - check your directory: {path}")

    newest_match = max(all_models, key=lambda x: x.stat().st_mtime)
    return newest_match


def _get_estimator_name(clf) -> str:
    """
    Returns estimator name based on class name. If passed classifier is a :class:
    `~sklearn.pipeline.Pipeline`, assume last step is the estimator and return that classes name

    Parameters
    ----------
    clf: BaseEstimator, Pipeline

    Returns
    -------
    str
        Name of estimator
    """
    class_name = clf.__class__.__name__

    if class_name == "Pipeline":
        return clf.steps[-1][1].__class__.__name__

    return class_name


def listify(collection) -> list:
    """
    Takes a given collection and returns a list of the elements, handling strings correctly

    Parameters
    ----------
    collection: tuple, set, str
        Any type of collection or string

    Returns
    -------
    list
    """
    if isinstance(collection, str):
        collection = [collection]

    if isinstance(collection, (tuple, set)):
        collection = list(collection)

    return collection


def _create_param_grid(pipe: Pipeline, param_grid: dict) -> ParameterGrid:
    """
    Creates a parameter grid from a :class:`~sklearn.pipeline.Pipeline`

    Parameters
    ----------
    pipe: Pipeline
        Input pipeline
    param_grid: dict
        dict of parameters to search over

    Returns
    -------
    :class:`~sklearn.model_selection.ParameterGrid`
    """
    if not isinstance(pipe, Pipeline):
        return ParameterGrid(param_grid)

    step_name = pipe.steps[-1][0]

    step_dict = {
        f"{step_name}__{param}" if step_name not in param else param: value
        for param, value in param_grid.items()
    }

    return ParameterGrid(step_dict)


def _validate_estimator(estimator):
    """
    Ensures that estimator is a valid estimator - either a :class:`~sklearn.base.BaseEstimator`
    or a :class:`~sklearn.pipeline.Pipeline` with a :class:`~sklearn.base.BaseEstimator`
    as the final step

    Parameters
    ----------
    estimator: passed estimator to validate

    Returns
    -------
    :class:`~sklearn.base.BaseEstimator`

    Raises
    ------
    MLToolingError
        Raises on invalid input
    """

    if hasattr(estimator, "_estimator_type"):
        return estimator

    if isinstance(estimator, Pipeline):
        raise MLToolingError(
            "You passed a Pipeline without an estimator as the last step"
        )

    raise MLToolingError(f"Expected a Pipeline or Estimator - got {type(estimator)}")
