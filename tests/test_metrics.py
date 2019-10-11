import numpy as np
import pandas as pd
import pytest

from ml_tooling.metrics import (
    lift_score,
    confusion_matrix,
    target_correlation,
    Metric,
    Metrics,
)
from ml_tooling.metrics.utils import (
    MetricError,
    _sort_values,
    _get_top_n_idx,
    _get_bottom_n_idx,
)


class TestMetricClass:
    def test_can_score_metric(self, classifier, test_dataset):
        metric = Metric("accuracy")
        metric.score_metric(classifier.estimator, test_dataset.x, test_dataset.y)
        assert metric.name == "accuracy"
        assert isinstance(metric.score, float)

        assert metric.cross_val_scores is None
        assert metric.std is None

    def test_can_score_metric_cv(self, classifier, test_dataset):
        metric = Metric("accuracy")
        metric.score_metric_cv(
            classifier.estimator,
            test_dataset.train_x,
            test_dataset.train_y,
            cv=2,
            n_jobs=-1,
            verbose=0,
        )
        assert metric.name == "accuracy"
        assert len(metric.cross_val_scores) == 2
        assert metric.score == np.mean(metric.cross_val_scores)
        assert metric.std == np.std(metric.cross_val_scores)


class TestMetricsClass:
    def test_can_list_metrics(self):
        metric_names = ["accuracy", "roc_auc"]
        metrics = Metrics.from_list(metric_names)
        assert metric_names == metrics.to_list()

    def test_can_access_attributes_from_underlying_metrics(self):
        metric_names = ["accuracy", "roc_auc"]
        metrics = Metrics.from_list(metric_names)

        assert metrics.name == "accuracy"
        with pytest.raises(AttributeError):
            metrics.not_an_attribute

    def test_can_create_from_dict(self):
        metrics = Metrics.from_list(["accuracy", "roc_auc"])
        metrics_dict = metrics.to_dict()
        metrics2 = Metrics.from_dict(metrics_dict)

        assert metrics == metrics2

    def test_scoring_multiple_metrics_works_correctly(self, classifier, test_dataset):
        accuracy = Metric("accuracy")
        roc_auc = Metric("roc_auc")
        metrics = Metrics.from_list(["accuracy", "roc_auc"])

        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert ("not_a_metric" in metrics) is False

        for metric in [accuracy, roc_auc]:
            metric.score_metric(classifier.estimator, test_dataset.x, test_dataset.y)

        metrics.score_metrics(classifier.estimator, test_dataset.x, test_dataset.y)

        assert metrics[0] == accuracy
        assert metrics[1] == roc_auc

        for metric in [accuracy, roc_auc]:
            metric.score_metric_cv(
                estimator=classifier.estimator,
                x=test_dataset.x,
                y=test_dataset.y,
                cv=2,
                n_jobs=-1,
                verbose=0,
            )
        metrics.score_metrics_cv(
            estimator=classifier.estimator,
            x=test_dataset.x,
            y=test_dataset.y,
            cv=2,
            n_jobs=-1,
            verbose=0,
        )
        assert (metrics[0].name, metrics[0].score) == (accuracy.name, accuracy.score)
        assert (metrics[1].name, metrics[1].score) == (roc_auc.name, roc_auc.score)
        assert all(metrics[0].cross_val_scores == accuracy.cross_val_scores)
        assert all(metrics[1].cross_val_scores == roc_auc.cross_val_scores)


class TestLiftScore:
    @pytest.mark.parametrize(
        "y_targ, y_pred",
        [
            (
                np.array([[1, 1, 0, 0], [0, 1, 0, 1]]),
                np.array([[1, 1, 0, 0], [1, 1, 1, 1]]),
            ),
            ([[1, 1, 0, 0], [0, 1, 0, 1]], [[1, 1, 0, 0], [1, 1, 1, 1]]),
        ],
    )
    def test_lift_score_fails_if_passed_non_ndarray_or_series(self, y_targ, y_pred):
        with pytest.raises(MetricError):
            # noinspection PyTypeChecker
            lift_score(y_targ, y_pred)

    @pytest.mark.parametrize(
        "y_targ, y_pred",
        [
            (
                np.array([1, 1, 1, 0, 0, 2, 0, 3, 4]),
                np.array([1, 0, 1, 0, 0, 2, 1, 3, 0]),
            ),
            (
                pd.Series([1, 1, 1, 0, 0, 2, 0, 3, 4]),
                pd.Series([1, 0, 1, 0, 0, 2, 1, 3, 0]),
            ),
            ((1, 1, 1, 0, 0, 2, 0, 3, 4), (1, 0, 1, 0, 0, 2, 1, 3, 0)),
            ([1, 1, 1, 0, 0, 2, 0, 3, 4], [1, 0, 1, 0, 0, 2, 1, 3, 0]),
        ],
    )
    def test_lift_score_returns_correctly(self, y_targ, y_pred):
        result = lift_score(y_targ, y_pred)
        assert 2 == result


class TestConfusionMatrix:
    def test_normalized_confusion_matrix_between_0_and_1(self):
        cm = confusion_matrix(
            np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), normalized=True
        )
        assert (cm >= 0).all() & (cm <= 1).all()
        assert 1 == np.sum(cm)

    def test_confusion_matrix_returns_as_expected(self):
        cm = confusion_matrix(
            np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1]), normalized=False
        )
        assert np.all(np.array([[0, 1], [0, 3]]) == cm)
        assert 4 == np.sum(cm)


class TestFeatureImportance:
    def test_sorted_feature_importance_returns_as_expected(self):
        labels = np.array(["Feature 1", "Feature 2"])
        importance = np.array([0.8, 0.7])

        result_labels, result_importance = _sort_values(labels, importance)
        assert np.all(labels == result_labels)
        assert np.all(importance == result_importance)

    def test_sorted_feature_importance_top_n_int_returns_as_expected(self):
        labels = np.array(
            ["Feature 1", "Features 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result_labels, result_importance = _sort_values(labels, importance, top_n=2)
        assert ["Feature 5", "Feature 4"] == list(result_labels)
        assert [0.5, 0.4] == list(result_importance)

    def test_sorted_feature_importance_top_n_percent_returns_as_expected(self):
        labels = np.array(
            ["Feature 1", "Features 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result_labels, result_importance = _sort_values(labels, importance, top_n=0.2)
        assert ["Feature 5"] == list(result_labels)
        assert [0.5] == list(result_importance)

    def test_sorted_feature_importance_bottom_n_int_returns_as_expected(self):
        labels = np.array(
            ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result_labels, result_importance = _sort_values(labels, importance, bottom_n=2)
        assert ["Feature 2", "Feature 1"] == list(result_labels)
        assert [0.2, 0.1] == list(result_importance)

    def test_sorted_feature_importance_bottom_n_percent_returns_as_expected(self):
        labels = np.array(
            ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result_labels, result_importance = _sort_values(
            labels, importance, bottom_n=0.2
        )
        assert ["Feature 1"] == list(result_labels)
        assert [0.1] == list(result_importance)

    def test_sorted_feature_importance_bottom_and_top_n_int_returns_as_expected(self):
        labels = np.array(
            ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result_labels, result_importance = _sort_values(
            labels, importance, bottom_n=2, top_n=1
        )
        assert ["Feature 5", "Feature 2", "Feature 1"] == list(result_labels)
        assert [0.5, 0.2, 0.1] == list(result_importance)

    def test_sorted_feature_importance_bottom_and_top_n_percent_returns_as_expected(
        self
    ):
        labels = np.array(
            ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result_labels, result_importance = _sort_values(
            labels, importance, bottom_n=0.2, top_n=0.2
        )
        assert ["Feature 5", "Feature 1"] == list(result_labels)
        assert [0.5, 0.1] == list(result_importance)

    def test_sorted_feature_importance_sorts_by_absolute_value_correctly(self):
        labels = np.array(
            ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        )
        importance = np.array([0.1, 0.2, 0.3, -0.4, -0.5])

        result_labels, result_importance = _sort_values(
            labels, importance, abs_sort=True
        )

        assert np.all(result_importance == np.array([-0.5, -0.4, 0.3, 0.2, 0.1]))


class TestUtils:
    @pytest.mark.parametrize(
        "n, expected",
        [(2, np.array([20, 10])), (0.1, np.array([20])), (0.05, np.array([20]))],
    )
    def test_top_n_is_correct_when_given_int_and_float(self, n, expected):
        input_array = np.array([20, 10, 5, 4, 2, 2, 1, 1, 0, 0])
        result = _get_top_n_idx(input_array, n)
        assert np.all(expected == result)

    @pytest.mark.parametrize(
        "n, expected",
        [(2, np.array([0, 0])), (0.1, np.array([0])), (0.05, np.array([0]))],
    )
    def test_bottom_n_is_correct_when_given_int_and_float(self, n, expected):
        input_array = np.array([20, 10, 5, 4, 2, 2, 1, 1, 0, 0])
        result = _get_bottom_n_idx(input_array, n)
        assert np.all(expected == result)


class TestCorrelation:
    def test_correlation_gives_expected_result(self):
        x_values = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        corr = target_correlation(x_values, y_values)
        assert corr.at["col1"] == 1.0

    def test_anscombes_quartet_gives_expected_result(self):
        x_values = pd.DataFrame(
            {
                "col1": [
                    8.04,
                    6.95,
                    7.58,
                    8.81,
                    8.33,
                    9.96,
                    7.24,
                    4.26,
                    10.84,
                    4.82,
                    5.68,
                ],
                "col2": [
                    9.14,
                    8.14,
                    8.74,
                    8.77,
                    9.26,
                    8.10,
                    6.13,
                    3.10,
                    9.13,
                    7.26,
                    4.74,
                ],
                "col3": [
                    7.46,
                    6.77,
                    12.74,
                    7.11,
                    7.81,
                    8.84,
                    6.08,
                    5.39,
                    8.15,
                    6.42,
                    5.73,
                ],
            }
        )
        y_values = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
        corr = target_correlation(x_values, y_values)
        assert (corr.round(3) == 0.816).all()
