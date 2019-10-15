import sys
import pytest
import joblib
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from artifactory import PureArtifactoryPath
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

    def mock_deploy_file(*args, **kwargs):
        return open(file_path, "wb").write(open_estimator_pickle.read())

    mock_artifactory_path.return_value.deploy_file = mock_deploy_file

    storage = ArtifactoryStorage("http://www.testy.com", "/test")
    storage.save("test", "test")
    f = joblib.load(file_path)
    assert isinstance(f, (BaseEstimator, Pipeline))


@require_artifactory
@patch("ml_tooling.storage.artifactory.ArtifactoryPath.glob")
@patch("ml_tooling.storage.artifactory.ArtifactoryPath")
def test_can_get_list_of_paths(mock_artifactory_path, mock_purepath_glob):
    url = "http://artifactory-singlep.p001.alm.brand.dk/artifactory/advanced-analytics/dev/"

    paths = [
        f"{url}LogisticRegression_2019-10-15_10:42:10.709197.pkl",
        f"{url}LogisticRegression_2019-10-15_10:32:41.780990.pkl",
        f"{url}LogisticRegression_2019-10-15_10:34:34.226695.pkl",
        f"{url}LogisticRegression_2019-10-15_10:51:50.760746.pkl",
        f"{url}LogisticRegression_2019-10-15_10:34:21.849358.pkl",
    ]
    mock_artifactory_path.return_value = mock_artifactory_path
    mock_purepath_glob.return_value = [PureArtifactoryPath(path) for path in paths]

    storage = ArtifactoryStorage("test", "test")
    artifactory_paths = storage.get_list()
    assert str(artifactory_paths[0]) == paths[1]
    assert str(artifactory_paths[-1]) == paths[3]


@require_artifactory
@patch("ml_tooling.storage.artifactory.ArtifactoryPath", autospec=True)
def test_artifactory_initialization_path(
    mock_artifactory_path, open_estimator_pickle, classifier
):
    mock_artifactory_path.return_value = mock_artifactory_path
    mock_artifactory_path.__str__.return_value = "http://www.testy.com"

    filename = "estimator.pkl"
    storage = ArtifactoryStorage("http://www.testy.com", "/test", apikey="key")
    storage.save(classifier.estimator, filename)
    mock_artifactory_path.assert_called_with(
        "http://www.testy.com/test/dev", apikey="key", auth=None
    )

    mock_artifactory_path.return_value.open.return_value = open_estimator_pickle
    storage.load("estimator.pkl")
    mock_artifactory_path.assert_called_with(
        "http://www.testy.com/test/estimator.pkl", apikey="key", auth=None
    )
