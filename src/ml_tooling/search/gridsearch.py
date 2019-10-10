from sklearn import clone
from sklearn.model_selection import ParameterGrid


def prepare_gridsearch_estimators(estimator, params):
    baseline_estimator = clone(estimator)
    grid = ParameterGrid(params)
    yield from (baseline_estimator.set_params(**params) for params in grid)
