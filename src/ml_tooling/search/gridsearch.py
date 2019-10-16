from typing import Iterator

from sklearn import clone
from sklearn.model_selection import ParameterGrid

from ml_tooling.utils import Estimator


def prepare_gridsearch_estimators(
    estimator: Estimator, params: dict
) -> Iterator[Estimator]:
    baseline_estimator = clone(estimator)
    grid = ParameterGrid(params)
    yield from (baseline_estimator.set_params(**params) for params in grid)
