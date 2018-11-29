import pathlib
from typing import Union, Callable

import numpy as np
import pandas as pd
import math
from git import Repo, InvalidGitRepositoryError
from sklearn.base import BaseEstimator
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
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state, resample

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


def _generate_sample_indices(random_state, n_samples):
    """
    Sample indices to be used for permutations
    Parameters
    ----------
    random_state : None, int, RandomState
        Seed or RandomsState instance for number generator
    n_samples : int
        Size of data to generate indices for

    Returns numpy.ndarray
    -------

    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices


def _find_unsampled_indices(sample_indices, n_samples):
    """
    Finds indices not in sample_indices. To be used for Out-Of-Bag
    Parameters
    ----------
    sample_indices : numpy.ndarray
        numpy.ndarray with indices
    n_samples : int
        Size of data to generate indices for.

    Returns numpy.ndarray
    -------

    """
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def _greater_score_is_better(scorer):
    """
    Determines if a higher score for a given scorer is better or worse

    Parameters
    ----------
    scorer:
        scikit-learn scorer

    Returns
    -------
    bool

    """
    return True if scorer._sign > 0 else False


def _permutation_importances(model, scorer, x, y, samples, seed=1337):
    """

    Parameters
    ----------
    model :
        a trained estimator exposing a predict or predict_proba method depending on the metric

    scorer : _ThresholdScorer or _PredictScorer
        sklearn scorer

    x : DataFrame
        Feature data

    y : DateSeries
        Target data

    samples : None, int, float

        None - Original data set i used. Not recommended for small data sets

        float - A new smaller data set is made from resampling with
                replacement form the original data set. Not recommended for small data sets.
                Recommended for very large data sets.

        Int - A new  data set is made from resampling with replacement form the original data.
              samples sets the number of resamples. Recommended for small data sets
               to ensure stable estimates of feature importance.

    seed : int
        Seed for random number generator for permutation.

    Returns
    -------
    np.array
        Decrease in score when permuting features
    float
        Baseline score without permutation

    """

    random_state = check_random_state(seed=seed)
    x = x.copy()
    y = y.copy()

    if samples is not None and \
            not (isinstance(samples, int) and samples > 0) and \
            not (isinstance(samples, float) and 0 < samples < 1):
        raise MLToolingError("samples must be None, float or int.")

    if samples:
        if _is_percent(samples):
            samples = math.floor(samples * len(x)) or 1
        x, y = resample(x, y, n_samples=samples, replace=True, random_state=random_state)

    baseline = scorer(model, x, y)

    imp = []

    sign = 1 if _greater_score_is_better(scorer) else -1

    for col in x.columns:
        save = x[col].copy()
        x[col] = random_state.permutation(x[col])
        m = scorer(model, x, y)
        x[col] = save
        drop_in_score = sign * (baseline - m)
        imp.append(drop_in_score)

    return np.array(imp), baseline
