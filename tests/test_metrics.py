import numpy as np
import pytest
from ml_tooling.metrics import lift_score, confusion_matrix, sorted_feature_importance
from ml_tooling.metrics import MetricError


class TestLiftScore:
    def test_lift_score_fails_if_passed_non_ndarray(self):
        with pytest.raises(MetricError):
            # noinspection PyTypeChecker
            lift_score([1, 2, 3], [4, 5, 6])

    def test_lift_score_returns_correctly(self):
        y_targ = np.array([1, 1, 1, 0, 0, 2, 0, 3, 4])
        y_pred = np.array([1, 0, 1, 0, 0, 2, 1, 3, 0])

        result = lift_score(y_targ, y_pred)
        assert 2 == result


class TestConfusionMatrix:
    def test_normalized_confusion_matrix_between_0_and_1(self):
        cm = confusion_matrix(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), normalized=True)
        assert (cm >= 0).all() & (cm <= 1).all()
        assert 1 == np.sum(cm)

    def test_confusion_matrix_returns_as_expected(self):
        cm = confusion_matrix(np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1]), normalized=False)
        assert np.all(np.array([[0, 1], [0, 3]]) == cm)
        assert 4 == np.sum(cm)


class TestFeatureImportance:
    def test_sorted_feature_importance_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2'])
        importance = np.array([.8, .7])

        result_labels, result_importance = sorted_feature_importance(labels, importance)
        assert np.all(labels == result_labels)
        assert np.all(importance == result_importance)

    def test_sorted_feature_importance_top_n_int_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Features 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = sorted_feature_importance(labels, importance, top_n=2)
        assert ['Feature 5', 'Feature 4'] == list(result_labels)
        assert [.5, .4] == list(result_importance)

    def test_sorted_feature_importance_top_n_percent_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Features 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = sorted_feature_importance(labels, importance, top_n=.2)
        assert ['Feature 5'] == list(result_labels)
        assert [.5] == list(result_importance)

    def test_sorted_feature_importance_bottom_n_int_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = sorted_feature_importance(labels, importance, bottom_n=2)
        assert ['Feature 1', 'Feature 2'] == list(result_labels)
        assert [.1, .2] == list(result_importance)

    def test_sorted_feature_importance_bottom_n_percent_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = sorted_feature_importance(labels, importance,
                                                                     bottom_n=.2)
        assert ['Feature 1'] == list(result_labels)
        assert [.1] == list(result_importance)

    def test_sorted_feature_importance_bottom_and_top_n_int_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = sorted_feature_importance(labels,
                                                                     importance,
                                                                     bottom_n=2,
                                                                     top_n=1)
        assert ['Feature 5', 'Feature 1', 'Feature 2'] == list(result_labels)
        assert [.5, .1, .2] == list(result_importance)

    def test_sorted_feature_importance_bottom_and_top_n_percent_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = sorted_feature_importance(labels,
                                                                     importance,
                                                                     bottom_n=.2,
                                                                     top_n=.2)
        assert ['Feature 5', 'Feature 1'] == list(result_labels)
        assert [.5, .1] == list(result_importance)
