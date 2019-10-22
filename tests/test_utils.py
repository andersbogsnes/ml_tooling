import pathlib
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_tooling.metrics.utils import _is_percent
from ml_tooling.plots.utils import _generate_text_labels

from ml_tooling.utils import get_git_hash, MLToolingError, _validate_estimator, make_dir


def test_get_git_hash_returns_correctly():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert 10 < len(git_hash)


@patch("ml_tooling.utils.subprocess")
def test_get_git_hash_returns_empty_if_git_not_found(mock_subprocess):
    mock_subprocess.check_output.side_effect = OSError
    git_hash = get_git_hash()
    assert git_hash == ""

    mock_subprocess.check_output.assert_called_with(["git", "rev-parse", "HEAD"])


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
