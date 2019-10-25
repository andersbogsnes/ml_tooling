import pathlib
import sys
import pytest
import joblib
from unittest.mock import MagicMock, patch

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from ml_tooling.storage.artifactory import ArtifactoryStorage
from ml_tooling.utils import MLToolingError

require_artifactory = pytest.mark.skipif(
    "artifactory" not in sys.modules, reason="artifactory must be installed"
)


@require_artifactory
def test_can_load_from_artifactory(open_estimator_pickle):
    mock_open = MagicMock()
    mock_open.open.return_value.__enter__.return_value = open_estimator_pickle

    mock_path = MagicMock()
    mock_path.__truediv__.return_value = mock_open

    storage = ArtifactoryStorage("testy", "test")
    storage.artifactory_path = mock_path
    f = storage.load("test")
    assert isinstance(f, (BaseEstimator, Pipeline))


@require_artifactory
def test_can_save_to_artifactory(open_estimator_pickle, tmp_path: pathlib.Path):
    file_path = tmp_path.joinpath("temp.pkl")

    def mock_deploy_file(*args, **kwargs):
        return file_path.write_bytes(open_estimator_pickle.read())

    mock = MagicMock()
    mock.__truediv__.deploy_file.return_value = mock_deploy_file()

    storage = ArtifactoryStorage("http://www.testy.com", "/test")
    storage.artifactory_path = mock
    storage.save("test", "test")
    f = joblib.load(file_path)
    assert isinstance(f, (BaseEstimator, Pipeline))


@require_artifactory
def test_can_get_list_of_paths():
    url = "http://artifactory-singlep.p001.alm.brand.dk/artifactory/advanced-analytics/dev/"

    paths = [
        f"{url}LogisticRegression_2019-10-15_10:42:10.709197.pkl",
        f"{url}LogisticRegression_2019-10-15_10:32:41.780990.pkl",
        f"{url}LogisticRegression_2019-10-15_10:34:34.226695.pkl",
        f"{url}LogisticRegression_2019-10-15_10:51:50.760746.pkl",
        f"{url}LogisticRegression_2019-10-15_10:34:21.849358.pkl",
    ]
    mock = MagicMock()
    mock.glob.return_value = paths

    storage = ArtifactoryStorage("test", "test")
    storage.artifactory_path = mock
    artifactory_paths = storage.get_list()
    assert str(artifactory_paths[0]) == paths[1]
    assert str(artifactory_paths[-1]) == paths[3]


@require_artifactory
def test_artifactory_initialization_path():
    from dohq_artifactory.auth import XJFrogArtApiAuth

    storage = ArtifactoryStorage("http://www.testy.com", "test", apikey="key")

    assert storage.artifactory_path.repo == "test"
    assert storage.artifactory_path.drive == "http://www.testy.com/artifactory"
    assert isinstance(storage.artifactory_path.auth, XJFrogArtApiAuth)
    assert storage.artifactory_path.auth.apikey == "key"


@require_artifactory
def test_artifactory_initialization_with_artifactory_suffix_works_as_expected():
    storage = ArtifactoryStorage("http://www.testy.com/artifactory", "test")
    assert storage.artifactory_path.repo == "test"
    assert storage.artifactory_path.drive == "http://www.testy.com/artifactory"
    assert storage.artifactory_path.auth is None


@patch("ml_tooling.storage.artifactory._has_artifactory")
def test_using_artifactory_without_installed_fails(mock_const):
    mock_const.__bool__.return_value = False
    with pytest.raises(
        MLToolingError,
        match="Artifactory not installed - run pip install dohq-artifactory",
    ):
        ArtifactoryStorage("test", "test")
