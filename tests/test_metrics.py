import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import get_scorer
from ml_tooling.utils import (MLToolingError,
                              )
from ml_tooling.metrics import (MetricError,
                                lift_score,
                                confusion_matrix,
                                sorted_feature_importance,
                                _permutation_importances,
                                )


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

    def test_permutation_importances_raises(self, regression):
        x = regression.data.train_x
        y = regression.data.train_y
        model = regression.model
        scorer = get_scorer(regression.default_metric)

        with pytest.raises(MLToolingError, match="samples must be None, float or int."):
            _, _ = _permutation_importances(model, scorer, x, y, '1', seed=1337)

    @pytest.mark.parametrize('setting, expected_importance, expected_baseline', [
        (None, np.array([0.00273266410, 0.413660488, 0.779916893, 0.6152784471]), 0.2671288886),
        (0.5, np.array([0.006008307, 0.4534291900, 1.042080126, 0.928642803]), 0.36053665),
        (1000, np.array([0.001367147, 0.3810664646, 0.70650115542, 0.91687247998]), 0.24313681138)

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
        assert pytest.approx(baseline, expected_baseline)

    def test_permutation_importances_works_as_expected_with_pipeline(self, base, pipeline_logistic):
        pipe = base(pipeline_logistic)
        pipe.score_model()
        x = pipe.data.train_x
        y = pipe.data.train_y
        model = pipe.model
        scorer = get_scorer(pipe.default_metric)
        importance, baseline = _permutation_importances(model, scorer, x, y, 1000, seed=1337)
        expected_importance = np.array([-0.0180000000, 0.171000000000, 0.051000000, 0.075999999999])
        expected_baseline = 0.759

        np.testing.assert_almost_equal(importance, expected_importance)
        assert pytest.approx(baseline, expected_baseline)

    def test_permutation_importances_works_with_proba_scorer(self, base, pipeline_logistic):
        pipe = base(pipeline_logistic)
        pipe.default_metric = 'roc_auc'
        pipe.score_model()
        x = pipe.data.train_x
        y = pipe.data.train_y
        model = pipe.model
        scorer = get_scorer(pipe.default_metric)
        importance, baseline = _permutation_importances(model, scorer, x, y, 1000, seed=1337)
        expected_importance = np.array([0.0072557875, 0.2974636312, 0.0944462426, 0.0781050511])
        expected_baseline = 0.8305146463829

        np.testing.assert_almost_equal(importance, expected_importance)
        assert pytest.approx(baseline, expected_baseline)

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
