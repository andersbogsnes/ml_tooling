import math
from typing import Union
from copy import deepcopy
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn import metrics
from sklearn.utils import check_random_state, resample
from .utils import (_is_percent,
                    DataType,
                    _greater_score_is_better,
                    MLToolingError,
                    )


class MetricError(Exception):
    pass


def lift_score(y_target: DataType, y_predicted: DataType) -> float:
    """
    Calculates lift score for a given model. The lift score quantifies how much better
    the model is compared to a random baseline.

    The formula is defined as follows:
        lift = (TP/(TP+FN)(TP+FP)/(TP+TN+FP+FN)
        Source: https://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score

    :param y_target:
        Target labels

    :param y_predicted:
        Predicted labels

    :return:
        Lift score
    """
    y_target = np.array(y_target)
    y_predicted = np.array(y_predicted)

    if y_target.ndim > 1 or y_predicted.ndim > 1:
        raise MetricError("Input must be 1-dimensional")

    n = len(y_target)
    percent_positives_target = np.sum(y_target == 1) / n
    percent_positives_predicted = np.sum(y_predicted == 1) / n

    all_prod = np.column_stack([y_target, y_predicted])
    percent_correct_positives = (all_prod == 1).all(axis=1).sum() / n

    return percent_correct_positives / (percent_positives_target * percent_positives_predicted)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalized=True) -> np.ndarray:
    """
    Generates a confusion matrix using sklearn's confusion_matrix function, with the ability
    to add normalization.

    Calculates a matrix of
        [[True Positives, False Positives],
         [False Negatives, True Negatives]]

    :param y_true:
        True labels

    :param y_pred:
        Predicted labels
    :param normalized:
        Whether or not to normalize counts

    :return:
        Confusion matrix array
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalized is True:
        cm = cm / cm.sum()
        cm = np.around(cm, 2)
        cm[np.isnan(cm)] = 0.0
    return cm


def _get_column_importance(model, scorer, x, y, seed, col):
    """
    Helper function for _permutation_importances to calculate the importance of a single column.
    When col=None the function calculates the baseline.

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

    seed :
        Seed for random number generator for permutation.

    col : str
        Which column to permute.

    Returns
    -------

    """
    if col:
        random_state = check_random_state(seed=seed)
        save = x[col].copy()
        x[col] = random_state.permutation(x[col])
    measure = scorer(model, x, y)
    if col:
        x[col] = save
    return measure


def _permutation_importances(model, scorer, x, y, samples, seed=1337, n_jobs=1):
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
    model = deepcopy(model)

    if samples is not None and \
            not (isinstance(samples, int) and samples > 0) and \
            not (isinstance(samples, float) and 0 < samples < 1):
        raise MLToolingError("samples must be None, float or int.")

    if samples:
        if _is_percent(samples):
            samples = math.floor(samples * len(x)) or 1
        x, y = resample(x, y, n_samples=samples, replace=True, random_state=random_state)

    # Used to ensure random number generation is independent of parallelization
    col_seeds = [None] + [i for i in range(seed, seed + len(x.columns))]
    cols = [None] + x.columns.tolist()

    measure = Parallel(n_jobs=n_jobs)(
        delayed(_get_column_importance)(model, scorer, x, y, col_seed, col) for col, col_seed in
        zip(cols, col_seeds))

    baseline = measure[0]
    sign = 1 if _greater_score_is_better(scorer) else -1
    drop_in_score = sign * (baseline - measure[1:])

    return np.array(drop_in_score), baseline


def sorted_feature_importance(labels: np.ndarray,
                              importance: np.ndarray,
                              top_n: Union[int, float] = None,
                              bottom_n: Union[int, float] = None):
    """
    Calculates a sorted array of importances and corresponding labels by absolute values

    :param labels:
        List of feature labels

    :param importance:
        List of importance values

    :param top_n:
        If top_n is an int return top n features
        If top_n is a float between 0 and 1 return top top_n percent of features

    :param bottom_n:
        If bottom_n is an int return bottom n features
        If bottom_n is a float between 0 and 1 return bottom bottom_n percent of features

    :return:
        List of labels and list of feature importances sorted by importance
    """
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    idx = np.argsort(importance)[::-1]

    sorted_idx = []

    if top_n:
        if _is_percent(top_n):
            top_n = math.floor(top_n * len(idx)) or 1  # If floor rounds to 0, use 1 instead
        sorted_idx.extend(idx[:top_n])

    if bottom_n:
        if _is_percent(bottom_n):
            bottom_n = math.floor(bottom_n * len(idx)) or 1  # If floor rounds to 0, use 1 instead
        sorted_idx.extend(idx[::-1][:bottom_n])

    if sorted_idx:
        idx = sorted_idx

    return labels[idx], importance[idx]
