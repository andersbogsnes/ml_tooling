import math
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_random_state, resample

from ml_tooling.metrics.utils import _is_percent
from ml_tooling.utils import MLToolingError, get_scoring_func


def _get_column_importance(model,
                           scorer,
                           x,
                           y,
                           seed,
                           col):
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


def _permutation_importances(model,
                             scorer,
                             x,
                             y,
                             samples,
                             seed=1337,
                             n_jobs=1,
                             verbose=0):
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

    measure = Parallel(n_jobs=n_jobs, verbose=verbose, max_nbytes=None)(
        delayed(_get_column_importance)(model, scorer, x, y, col_seed, col) for col, col_seed in
        zip(cols, col_seeds))

    baseline = measure[0]
    drop_in_score = baseline - measure[1:]

    return np.array(drop_in_score), baseline


def _get_feature_importance(viz, samples, seed=1337, n_jobs=1, verbose=0) -> pd.DataFrame:
    """
    Helper function for extracting importances.

    Parameters
    ----------
    viz : BaseVisualize
        An instance of BaseVisualizer

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
    model = viz._model
    scorer = get_scoring_func(viz.default_metric)
    train_x = viz._data.train_x.copy()
    train_y = viz._data.train_y.copy()

    return _permutation_importances(model,
                                    scorer,
                                    train_x,
                                    train_y,
                                    samples,
                                    seed,
                                    n_jobs,
                                    verbose)
