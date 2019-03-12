import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer

from ml_tooling.metrics import (lift_score,
                                confusion_matrix,
                                target_correlation
                                )
from ml_tooling.metrics.permutation_importance import _permutation_importances
from ml_tooling.metrics.utils import (MetricError,
                                      _sort_values,
                                      )
from ml_tooling.utils import MLToolingError


class TestLiftScore:
    @pytest.mark.parametrize('y_targ, y_pred', [
        (np.array([[1, 1, 0, 0], [0, 1, 0, 1]]), np.array([[1, 1, 0, 0], [1, 1, 1, 1]])),
        ([[1, 1, 0, 0], [0, 1, 0, 1]], [[1, 1, 0, 0], [1, 1, 1, 1]])
    ])
    def test_lift_score_fails_if_passed_non_ndarray_or_series(self, y_targ, y_pred):
        with pytest.raises(MetricError):
            # noinspection PyTypeChecker
            lift_score(y_targ, y_pred)

    @pytest.mark.parametrize('y_targ, y_pred', [
        (np.array([1, 1, 1, 0, 0, 2, 0, 3, 4]), np.array([1, 0, 1, 0, 0, 2, 1, 3, 0])),
        (pd.Series([1, 1, 1, 0, 0, 2, 0, 3, 4]), pd.Series([1, 0, 1, 0, 0, 2, 1, 3, 0])),
        ((1, 1, 1, 0, 0, 2, 0, 3, 4), (1, 0, 1, 0, 0, 2, 1, 3, 0)),
        ([1, 1, 1, 0, 0, 2, 0, 3, 4], [1, 0, 1, 0, 0, 2, 1, 3, 0])
    ])
    def test_lift_score_returns_correctly(self, y_targ, y_pred):
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

        result_labels, result_importance = _sort_values(labels, importance)
        assert np.all(labels == result_labels)
        assert np.all(importance == result_importance)

    def test_sorted_feature_importance_top_n_int_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Features 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = _sort_values(labels, importance, top_n=2)
        assert ['Feature 5', 'Feature 4'] == list(result_labels)
        assert [.5, .4] == list(result_importance)

    def test_sorted_feature_importance_top_n_percent_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Features 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = _sort_values(labels, importance, top_n=.2)
        assert ['Feature 5'] == list(result_labels)
        assert [.5] == list(result_importance)

    def test_sorted_feature_importance_bottom_n_int_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = _sort_values(labels, importance, bottom_n=2)
        assert ['Feature 1', 'Feature 2'] == list(result_labels)
        assert [.1, .2] == list(result_importance)

    def test_sorted_feature_importance_bottom_n_percent_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = _sort_values(labels, importance,
                                                        bottom_n=.2)
        assert ['Feature 1'] == list(result_labels)
        assert [.1] == list(result_importance)

    def test_sorted_feature_importance_bottom_and_top_n_int_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = _sort_values(labels,
                                                        importance,
                                                        bottom_n=2,
                                                        top_n=1)
        assert ['Feature 5', 'Feature 1', 'Feature 2'] == list(result_labels)
        assert [.5, .1, .2] == list(result_importance)

    def test_sorted_feature_importance_bottom_and_top_n_percent_returns_as_expected(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, .4, .5])

        result_labels, result_importance = _sort_values(labels,
                                                        importance,
                                                        bottom_n=.2,
                                                        top_n=.2)
        assert ['Feature 5', 'Feature 1'] == list(result_labels)
        assert [.5, .1] == list(result_importance)

    def test_sorted_feature_importance_sorts_by_absolute_value_correctly(self):
        labels = np.array(['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'])
        importance = np.array([.1, .2, .3, -.4, -.5])

        result_labels, result_importance = _sort_values(labels,
                                                        importance,
                                                        sort='abs')

        assert np.all(result_importance == np.array([-.5, -.4, .3, .2, .1]))

    def test_permutation_importances_raises(self, regression):
        x = regression.data.train_x
        y = regression.data.train_y
        model = regression.model
        scorer = get_scorer(regression.default_metric)

        with pytest.raises(MLToolingError, match="samples must be None, float or int."):
            _permutation_importances(model, scorer, x, y, '1', seed=1337)

    @pytest.mark.parametrize('setting, expected_importance, expected_baseline', [
        (None, np.array([0.00273266410, 0.4262626, 0.9134092, 1.1119456]), 0.2671288886),
        (0.5, np.array([0.0043651, 0.6032381, 1.113386, 0.7495175]), 0.36053665),
        (1000, np.array([0.0016836, 0.4391413, 0.8194372, 0.8254109]), 0.24313681138)

    ])
    def test_permutation_importances_works_as_expected_with_estimator(self, regression, setting,
                                                                      expected_importance,
                                                                      expected_baseline):
        x = regression.data.train_x
        y = regression.data.train_y
        model = regression.model
        scorer = get_scorer(regression.default_metric)
        importance, baseline = _permutation_importances(model, scorer, x, y, setting, seed=1337)

        np.testing.assert_almost_equal(importance, expected_importance)
        assert pytest.approx(baseline) == pytest.approx(expected_baseline)

    def test_permutation_importances_works_as_expected_with_pipeline(self, base, pipeline_logistic):
        pipe = base(pipeline_logistic)
        pipe.score_model()
        x = pipe.data.train_x
        y = pipe.data.train_y
        model = pipe.model
        scorer = get_scorer(pipe.default_metric)
        importance, baseline = _permutation_importances(model, scorer, x, y, 1000, seed=1337)
        expected_importance = np.array([-0.0190000000, 0.164000000000, 0.038000000, 0.0740])
        expected_baseline = 0.759

        np.testing.assert_almost_equal(importance, expected_importance)
        assert pytest.approx(baseline) == pytest.approx(expected_baseline)

    def test_permutation_importances_works_with_proba_scorer(self, base, pipeline_logistic):
        pipe = base(pipeline_logistic)
        pipe.default_metric = 'roc_auc'
        pipe.score_model()
        x = pipe.data.train_x
        y = pipe.data.train_y
        model = pipe.model
        scorer = get_scorer(pipe.default_metric)
        importance, baseline = _permutation_importances(model, scorer, x, y, 1000, seed=1337)
        expected_importance = np.array([0.0035604, 0.3021749, 0.1075911, 0.0688982])
        expected_baseline = 0.8305146463829

        np.testing.assert_almost_equal(importance, expected_importance)
        assert pytest.approx(baseline) == pytest.approx(expected_baseline)

    def test_permutation_importances_gives_same_result_in_parallel(self, base, pipeline_logistic):
        pipe = base(pipeline_logistic)
        pipe.score_model()
        x = pipe.data.train_x
        y = pipe.data.train_y
        model = pipe.model
        scorer = get_scorer(pipe.default_metric)
        importance_parellel, baseline_parellel = _permutation_importances(model, scorer, x, y, 100,
                                                                          seed=1337, n_jobs=-1)
        importance_single, baseline_single = _permutation_importances(model, scorer, x, y, 100,
                                                                      seed=1337, n_jobs=1)

        assert np.all(importance_parellel == importance_single)
        assert baseline_single == baseline_parellel


class TestCorrelation:
    def test_correlation_gives_expected_result(self):
        x_values = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        corr = target_correlation(x_values, y_values)
        assert corr.at["col1"] == 1.0

    def test_anscombes_quartet_gives_expected_result(self):
        x_values = pd.DataFrame(
            {"col1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
             "col2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
             "col3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]})
        y_values = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
        corr = target_correlation(x_values, y_values)
        assert (corr.round(3) == 0.816).all()
