from typing import Iterator

from sklearn import clone
from sklearn.model_selection import ParameterSampler

from ml_tooling.utils import Estimator


def prepare_randomsearch_estimators(
    estimator: Estimator, params: dict, n_iter: int = 10, random_state=1337
) -> Iterator[Estimator]:
    grid = ParameterSampler(params, n_iter=n_iter, random_state=random_state)
    yield from (clone(estimator).set_params(**p) for p in grid)
