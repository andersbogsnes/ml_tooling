import numpy as np
import pytest
from ml_tools.metrics import lift_score, confusion_matrix, sorted_feature_importance
from ml_tools.metrics.metrics import MetricError


# noinspection PyTypeChecker
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
    assert 1 == np.sum(cm)


def test_confusion_matrix_returns_as_expected():
    cm = confusion_matrix(np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1]), normalized=False)
    assert np.all(np.array([[0, 1], [0, 3]]) == cm)
    assert 4 == np.sum(cm)


def test_sorted_feature_importance_returns_as_expected():
    labels = np.array(['Feature 1', 'Feature 2'])
    importance = np.array([.8, .7])

    result_labels, result_importance = sorted_feature_importance(labels, importance)
    assert np.all(labels == result_labels)
    assert np.all(importance == result_importance)


def test_sorted_feature_importance_ascending_returns_as_expected():
    labels = np.array(['Feature 1', 'Feature 2'])
    importance = np.array([.8, .7])

    result_labels, result_importance = sorted_feature_importance(labels,
                                                                 importance,
                                                                 ascending=True)
    assert np.all(np.array(['Feature 2', 'Feature 1']) == result_labels)
    assert np.all(np.array([.7, .8]) == result_importance)
