import matplotlib.pyplot as plt
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_tooling.logging.log_estimator import _make_run_dir
from ml_tooling.metrics.utils import _is_percent
from ml_tooling.plots.utils import _generate_text_labels

from ml_tooling.utils import (
    get_git_hash,
    find_estimator_file,
    MLToolingError,
    _create_param_grid,
    _validate_estimator,
)


def test_get_git_hash_returns_correctly():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert 10 < len(git_hash)


def test_find_model_file_with_given_model_returns_correctly(tmpdir):
    model_folder = tmpdir.mkdir("estimator")
    model1 = "TestModel1_1234.pkl"
    model1_file = model_folder.join(model1)
    model1_file.write("test")

    model2 = "TestModel2_1234.pkl"
    model2_file = model_folder.join(model2)
    model2_file.write("test")

    result = find_estimator_file(model1_file)

    assert model1_file == result


def test_find_model_raise_when_no_model_found():
    with pytest.raises(
        MLToolingError, match="No models found - check your directory: nonsense"
    ):
        find_estimator_file("nonsense")


def test_find_model_file_if_multiple_with_same_hash(tmpdir, monkeypatch):
    def mockreturn():
        return "1234"

    monkeypatch.setattr("ml_tooling.utils.get_git_hash", mockreturn)

    model_folder = tmpdir.mkdir("estimator")
    model1 = "TestModel1_1234.pkl"
    model1_file = model_folder.join(model1)
    model1_file.write("test")
    first_file_mtime = model1_file.mtime()

    model2 = "TestModel2_1234.pkl"
    model2_file = model_folder.join(model2)
    model2_file.write("test")
    model2_file.setmtime(first_file_mtime + 100)  # Ensure second file is newer

    result = find_estimator_file(model_folder)

    assert model2_file == result


@pytest.mark.parametrize(
    "number, is_percent",
    [(0.2, True), (1, False), (10, False), (0.00000000001, True), (1000000, False)],
)
def test_is_percent_returns_correctly(number, is_percent):
    assert _is_percent(number) is is_percent


def test_is_percent_raises_correctly_if_given_large_float():
    with pytest.raises(
        ValueError, match="Floats only valid between 0 and 1. Got 100.0"
    ):
        _is_percent(100.0)


def test_add_text_labels_vertical_returns_correct():
    fig, ax = plt.subplots()
    ax.bar(["value"], [100])
    x_values, y_values = next(_generate_text_labels(ax, horizontal=False))
    assert 0 == x_values
    assert 100 == y_values


def test_add_text_labels_horizontal_returns_correct():
    fig, ax = plt.subplots()
    ax.barh(["value"], [100])
    x_values, y_values = next(_generate_text_labels(ax, horizontal=True))
    assert 0 == y_values
    assert 100 == x_values


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(),
        make_pipeline(StandardScaler(), RandomForestClassifier()),
    ],
)
def test_validate_estimator_should_return_estimator(estimator):
    result = _validate_estimator(estimator)
    assert result is estimator


def test_validate_estimator_should_raise_on_invalid_input():
    class AnyClass:
        def __str__(self):
            return "<AnyClass>"

    with pytest.raises(MLToolingError, match=f"Expected a Pipeline or Estimator - got"):
        _validate_estimator(AnyClass)

    with pytest.raises(
        MLToolingError,
        match="You passed a Pipeline without an estimator as the last step",
    ):
        _validate_estimator(make_pipeline(StandardScaler()))


class TestGridsearchParams:
    def test_create_gridsearch_params_in_pipeline_returns_correct(
        self, pipeline_forest_classifier
    ):
        param_grid = {"n_estimators": [5, 10, 20], "max_depth": [3, 4, 5]}
        grid = _create_param_grid(pipeline_forest_classifier, param_grid)

        assert [
            {"clf__n_estimators": [5, 10, 20], "clf__max_depth": [3, 4, 5]}
        ] == grid.param_grid

    def test_create_gridsearch_params_returns_if_already_prepended(
        self, pipeline_forest_classifier
    ):
        param_grid = {"clf__n_estimators": [5, 10, 20], "clf__max_depth": [3, 4, 5]}

        grid = _create_param_grid(pipeline_forest_classifier, param_grid)

        assert [param_grid] == grid.param_grid

    def test_create_gridsearch_params_without_pipeline_returns_correct(self):
        param_grid = {"n_estimators": [5, 10, 20], "max_depth": [3, 4, 5]}
        model = RandomForestClassifier()
        grid = _create_param_grid(model, param_grid)

        assert [param_grid] == grid.param_grid


def test__make_run_dir_fails_if_passed_file(tmpdir):
    new_file = tmpdir.mkdir("test").join("test.txt")
    new_file.write("test hi")
    with pytest.raises(IOError):
        _make_run_dir(str(new_file))
