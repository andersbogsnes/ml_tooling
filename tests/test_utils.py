import pathlib
import subprocess
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_tooling.metrics.utils import _is_percent
from ml_tooling.plots.utils import _generate_text_labels

from ml_tooling.utils import (
    get_git_hash,
    MLToolingError,
    _validate_estimator,
    make_dir,
    _find_src_dir,
    _find_setup_file,
)


def test_get_git_hash_returns_correctly():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert 10 < len(git_hash)


@patch("ml_tooling.utils.subprocess")
def test_get_git_hash_returns_empty_if_git_not_found(mock_subprocess):
    mock_subprocess.check_output.side_effect = OSError
    with pytest.warns(UserWarning, match="Error using git - is `git` installed?"):
        git_hash = get_git_hash()
    assert git_hash == ""

    mock_subprocess.check_output.assert_called_with(["git", "rev-parse", "HEAD"])


@patch("ml_tooling.utils.subprocess")
def test_get_git_hash_returns_empty_and_emits_warning_if_git_not_found(mock_subprocess):
    mock_subprocess.check_output.side_effect = subprocess.CalledProcessError(
        128, cmd=["git", "rev-parse", "HEAD"]
    )
    with pytest.warns(
        UserWarning,
        match="Error using git - skipping git hash. Did you call `git init`?",
    ):
        git_hash = get_git_hash()

    assert git_hash == ""


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

    with pytest.raises(MLToolingError, match="Expected a Pipeline or Estimator - got"):
        _validate_estimator(AnyClass)

    with pytest.raises(
        MLToolingError,
        match="You passed a Pipeline without an estimator as the last step",
    ):
        _validate_estimator(make_pipeline(StandardScaler()))


class TestGridsearchParams:
    def test_make_dir_fails_on_input_files(self, tmp_path: pathlib.Path):
        file_path = tmp_path / "test.txt"

        file_path.write_text("test\ntest")

        with pytest.raises(IOError):
            make_dir(file_path)

    def test_make_dir_creates_folder_on_input_files(self, tmp_path: pathlib.Path):
        file_path = tmp_path / "testing"
        assert file_path.exists() is False

        make_dir(file_path)

        assert file_path.exists()


class TestFindSrcDir:
    def test_can_find_setup_file_from_root(self, temp_project_structure):
        result = _find_setup_file(temp_project_structure, 0, 2)
        assert result == temp_project_structure

    def test_can_find_setup_file_from_inside_project(self, temp_project_structure):
        result = _find_setup_file(
            temp_project_structure / "src" / "my_test_project", 0, 3
        )
        assert result == temp_project_structure

    def test_can_find_setup_file_from_outside_project(
        self, temp_project_structure: pathlib.Path
    ):
        result = _find_setup_file(temp_project_structure / "notebooks", 0, 3)
        assert result == temp_project_structure

    def test_find_setup_file_errors_when_no_setup_file_found(
        self, tmp_path: pathlib.Path
    ):
        with pytest.raises(MLToolingError):
            _find_setup_file(tmp_path / "test" / "test", 0, 2)

    def test_can_find_src_dir_from_root_folder_structure(
        self, temp_project_structure: pathlib.Path
    ):
        result = _find_src_dir(temp_project_structure)
        assert result == temp_project_structure / "src" / "my_test_project"

    def test_can_find_src_dir_from_inside_project(
        self, temp_project_structure: pathlib.Path
    ):
        result = _find_src_dir(temp_project_structure / "notebooks")
        assert result == temp_project_structure / "src" / "my_test_project"

    def test_find_src_dir_errors_when_no_src_is_found(self, tmp_path: pathlib.Path):
        tmp_path.joinpath("setup.py").write_text("I exist")
        with pytest.raises(MLToolingError, match="Project must have a src folder!"):
            _find_src_dir(tmp_path / "test" / "test")

    def test_find_src_dir_errors_when_no_init_is_found(self, tmp_path: pathlib.Path):
        tmp_path.joinpath("setup.py").write_text("I exist")
        output_folder = tmp_path / "src" / "test"
        output_folder.mkdir(parents=True)
        with pytest.raises(
            MLToolingError,
            match=f"No modules found in {output_folder.parent}! "
            f"Is there an __init__.py file in your module?",
        ):
            _find_src_dir(output_folder)
