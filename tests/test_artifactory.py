import sys
import pytest
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from ml_tooling.storage.artifactory import ArtifactoryStorage

require_artifactory = pytest.mark.skipif(
    "artifactory" not in sys.modules, reason="artifactory must be installed"
)


@require_artifactory
@patch("ml_tooling.storage.artifactory.ArtifactoryPath", autospec=True)
def test_can_load_from_artifactory(mock_artifactory_path, open_estimator_pickle):
    mock_artifactory_path.return_value.open.return_value = open_estimator_pickle
    f = ArtifactoryStorage("testy", "test").load("test")
    assert isinstance(f, (BaseEstimator, Pipeline))


@require_artifactory
@patch("ml_tooling.storage.artifactory.ArtifactoryPath", autospec=True)
def test_can_save_to_artifactory(
    mock_artifactory_path, open_estimator_pickle, tmp_path
):
    file_path = tmp_path.joinpath("temp.pkl")
    mock_artifactory_path.return_value.deploy_file.return_value = open(
        file_path, "wb"
    ).write(open_estimator_pickle.read())
    mock_artifactory_path.return_value = mock_artifactory_path
    assert file_path.is_file()
