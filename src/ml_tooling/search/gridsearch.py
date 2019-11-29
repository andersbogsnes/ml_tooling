from typing import Iterator

from sklearn import clone
from sklearn.model_selection import ParameterGrid

from ml_tooling.utils import Estimator


def prepare_gridsearch_estimators(
    estimator: Estimator, params: dict
) -> Iterator[Estimator]:
    grid = ParameterGrid(params)
    return [clone(estimator).set_params(**p) for p in grid]
