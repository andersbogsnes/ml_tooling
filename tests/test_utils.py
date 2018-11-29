import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.scorer import (_PredictScorer,
                                    get_scorer,
                                    )
from ml_tooling.logging import _make_run_dir
from ml_tooling.plots import _generate_text_labels
from ml_tooling.utils import (get_git_hash,
                              find_model_file,
                              _is_percent,
                              MLToolingError,
                              get_scoring_func,
                              _create_param_grid,
                              _permutation_importances,
                              _greater_score_is_better)


def test_get_git_hash_returns_correctly():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert 10 < len(git_hash)


def test_find_model_file_with_given_model_returns_correctly(tmpdir):
    model_folder = tmpdir.mkdir('model')
    model1 = 'TestModel1_1234.pkl'
    model1_file = model_folder.join(model1)
    model1_file.write('test')

    model2 = 'TestModel2_1234.pkl'
    model2_file = model_folder.join(model2)
    model2_file.write('test')

    result = find_model_file(model1_file)

    assert model1_file == result


def test_find_model_raise_when_no_model_found():
    with pytest.raises(MLToolingError, match="No models found - check your directory: nonsense"):
        find_model_file('nonsense')


def test_find_model_file_if_multiple_with_same_hash(tmpdir, monkeypatch):
    def mockreturn():
        return '1234'

    monkeypatch.setattr('ml_tooling.utils.get_git_hash', mockreturn)

    model_folder = tmpdir.mkdir('model')
    model1 = 'TestModel1_1234.pkl'
    model1_file = model_folder.join(model1)
    model1_file.write('test')
    first_file_mtime = model1_file.mtime()

    model2 = 'TestModel2_1234.pkl'
    model2_file = model_folder.join(model2)
    model2_file.write('test')
    model2_file.setmtime(first_file_mtime + 100)  # Ensure second file is newer

    result = find_model_file(model_folder)

    assert model2_file == result


@pytest.mark.parametrize('number, is_percent', [
    (0.2, True),
    (1, False),
    (10, False),
    (.00000000001, True),
    (1000000, False)
])
def test_is_percent_returns_correctly(number, is_percent):
    assert _is_percent(number) is is_percent


def test_is_percent_raises_correctly_if_given_large_float():
    with pytest.raises(ValueError, match='Floats only valid between 0 and 1. Got 100.0'):
        _is_percent(100.0)


def test_scoring_func_returns_a_scorer(classifier):
    scorer = get_scoring_func('accuracy')

    score = scorer(classifier.model, classifier.data.test_x, classifier.data.test_y)
    assert isinstance(scorer, _PredictScorer)
    assert score > 0.63


def test_scoring_func_fails_if_invalid_scorer_is_given():
    with pytest.raises(MLToolingError):
        get_scoring_func('invalid_scorer')


def test_add_text_labels_vertical_returns_correct():
    fig, ax = plt.subplots()
    ax.bar(['value'], [100])
    x_values, y_values = next(_generate_text_labels(ax, horizontal=False))
    assert 0 == x_values
    assert (100 + 105 * .005) == y_values


def test_add_text_labels_horizontal_returns_correct():
    fig, ax = plt.subplots()
    ax.barh(['value'], [100])
    x_values, y_values = next(_generate_text_labels(ax, horizontal=True))
    assert 0 == y_values
    assert (100 + 105 * .005) == x_values


class TestGridsearchParams:
    def test_create_gridsearch_params_in_pipeline_returns_correct(self, pipeline_forest_classifier):
        param_grid = {"n_estimators": [5, 10, 20],
                      "max_depth": [3, 4, 5]}
        grid = _create_param_grid(pipeline_forest_classifier, param_grid)

        assert [{"clf__n_estimators": [5, 10, 20],
                 "clf__max_depth": [3, 4, 5]}] == grid.param_grid

    def test_create_gridsearch_params_returns_if_already_prepended(self,
                                                                   pipeline_forest_classifier):
        param_grid = {"clf__n_estimators": [5, 10, 20],
                      "clf__max_depth": [3, 4, 5]}

        grid = _create_param_grid(pipeline_forest_classifier, param_grid)

        assert [param_grid] == grid.param_grid

    def test_create_gridsearch_params_without_pipeline_returns_correct(self):
        param_grid = {"n_estimators": [5, 10, 20],
                      "max_depth": [3, 4, 5]}
        model = RandomForestClassifier()
        grid = _create_param_grid(model, param_grid)

        assert [param_grid] == grid.param_grid


def test__make_run_dir_fails_if_passed_file(tmpdir):
    new_file = tmpdir.mkdir('test').join('test.txt')
    new_file.write('test hi')
    with pytest.raises(IOError):
        _make_run_dir(str(new_file))


def test_permutation_importances_raises(regression):
    x = regression.data.train_x
    y = regression.data.train_y
    model = regression.model
    scorer = get_scorer(regression.default_metric)

    with pytest.raises(MLToolingError, match="samples must be None, float or int."):
        _, _ = _permutation_importances(model, scorer, x, y, '1', seed=1337)


@pytest.mark.parametrize('metric, expected', [
    ('r2', True),
    ('neg_mean_squared_error', False)

])
def test_greater_score_is_better(metric, expected):
    scorer = get_scorer(metric)
    assert _greater_score_is_better(scorer) == expected


@pytest.mark.parametrize('setting, expected_importance, expected_baseline', [
    (None, np.array([0.00273266410, 0.413660488, 0.779916893, 0.6152784471]), 0.2671288886),
    (0.5, np.array([0.006008307, 0.4534291900, 1.042080126, 0.928642803]), 0.36053665),
    (1000, np.array([0.001367147, 0.3810664646, 0.70650115542, 0.91687247998]), 0.24313681138)

])
def test_permutation_importances_works_as_expected_with_estimator(regression, setting,
                                                                  expected_importance,
                                                                  expected_baseline):
    x = regression.data.train_x
    y = regression.data.train_y
    model = regression.model
    scorer = get_scorer(regression.default_metric)
    importance, baseline = _permutation_importances(model, scorer, x, y, setting, seed=1337)

    np.testing.assert_almost_equal(importance, expected_importance)
    assert pytest.approx(baseline, expected_baseline)


def test_permutation_importances_works_as_expected_with_pipeline(base, pipeline_logistic):
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


def test_permutation_importances_works_with_proba_scorer(base, pipeline_logistic):
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
