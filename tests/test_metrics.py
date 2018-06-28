import numpy as np
import pytest
from ml_utils.metrics import lift_score, confusion_matrix
from ml_utils.metrics.metrics import MetricError


def test_lift_score_fails_if_passed_non_ndarray():
    with pytest.raises(MetricError):
        lift_score([1, 2, 3], [4, 5, 6])


def test_lift_score_returns_correctly():
    y_targ = np.array([1, 1, 1, 0, 0, 2, 0, 3, 4])
    y_pred = np.array([1, 0, 1, 0, 0, 2, 1, 3, 0])

    result = lift_score(y_targ, y_pred)
    assert 2 == result


def test_normalized_confusion_matrix_between_0_and_1():
    cm = confusion_matrix(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), normalized=True)
    assert (cm >= 0).all() & (cm <= 1).all()


def test_confusion_matrix_returns_as_expected():
    cm = confusion_matrix(np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1]), normalized=False)
    assert np.all(np.array([[0, 1], [0, 3]]) == cm)
