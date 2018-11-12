import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.scorer import _PredictScorer
from sklearn.pipeline import Pipeline

from ml_tooling.logging import _make_run_dir
from ml_tooling.plots import _generate_text_labels
from ml_tooling.transformers import ToCategorical
from ml_tooling.utils import (get_git_hash,
                              find_model_file,
                              _is_percent,
                              MLToolingError,
                              get_scoring_func,
                              _create_param_grid,
                              _get_labels)


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


class TestGetLabels:
    def test_get_labels_returns_correctly_in_pipeline(self, categorical):
        pipe = Pipeline([
            ('cat', ToCategorical()),
            ('clf', DummyClassifier())
        ])

        pipe.fit(categorical, [0, 1, 0, 1])
        labels = _get_labels(pipe, categorical)
        expected_cols = ['category_a_a1',
                         'category_a_a2',
                         'category_a_a3',
                         'category_b_b1',
                         'category_b_b2',
                         'category_b_b3']

        assert set(expected_cols) == set(labels)

    def test_get_labels_returns_array_if_there_are_no_columns(self, regression):
        test_x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        # noinspection PyTypeChecker
        labels = _get_labels(regression, test_x)
        assert np.all(np.arange(test_x.shape[1]) == labels)

    def test_get_labels_outside_pipeline_returns_correctly(self, regression):
        model = regression.model
        data = regression.data.train_x

        labels = _get_labels(model, data)
        expected_labels = {'sepal length (cm)',
                           'sepal width (cm)',
                           'petal length (cm)',
                           'petal width (cm)'}

        assert expected_labels == set(labels)


def test__make_run_dir_fails_if_passed_file(tmpdir):
    new_file = tmpdir.mkdir('test').join('test.txt')
    new_file.write('test hi')
    with pytest.raises(IOError):
        _make_run_dir(str(new_file))
