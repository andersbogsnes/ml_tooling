from sklearn import clone
from sklearn.model_selection import ParameterGrid
import joblib


def _fit_gridpoint(estimator, params, train_x, train_y):
    estimator.set_params(**params)
    estimator.fit(train_x, train_y)
    return estimator


def gridsearch(estimator, train_x, train_y, params, n_jobs):
    baseline_estimator = clone(estimator)
    grid = ParameterGrid(params)

    parallel = joblib.Parallel(n_jobs=n_jobs)
    return parallel(
        (
            joblib.delayed(_fit_gridpoint)(
                estimator=baseline_estimator,
                params=params,
                train_x=train_x,
                train_y=train_y,
            )
            for params in grid
        )
    )
