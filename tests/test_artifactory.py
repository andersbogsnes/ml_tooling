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
