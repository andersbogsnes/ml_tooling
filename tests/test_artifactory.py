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


class TestArtifactoryStorage:
    @require_artifactory
    @patch("ml_tooling.storage.artifactory.ArtifactoryPath")
    def test_can_load_from_artifactory(
        self, artifactorypath_mock, open_estimator_pickle
    ):
        artifactorypath_mock.return_value.is_file.return_value = False

        mock_open = MagicMock()
        mock_open.open = open_estimator_pickle

        mock_path = MagicMock()
        mock_path.__truediv__.return_value = mock_open

        storage = ArtifactoryStorage("http://www.testy.com/artifactory", "test")
        storage.artifactory_path = mock_path
        f = storage.load("test")

        assert isinstance(f, (BaseEstimator, Pipeline))

    @require_artifactory
    def test_can_save_to_artifactory(
        self, open_estimator_pickle, tmp_path: pathlib.Path, regression
    ):
        file_path = tmp_path.joinpath("temp.pkl")

        def mock_deploy_file(*args, **kwargs):
            return file_path.write_bytes(open_estimator_pickle().read())

        mock = MagicMock()
        mock.__truediv__.return_value.deploy_file = mock_deploy_file
        mock.is_file.return_value = False

        storage = ArtifactoryStorage("http://www.testy.com", "/test")
        storage.artifactory_path = mock
        storage.save(regression.estimator, "test")
        f = joblib.load(file_path)
        assert isinstance(f, (BaseEstimator, Pipeline))

    @require_artifactory
    @patch("ml_tooling.storage.artifactory.ArtifactoryPath")
    def test_can_get_list_of_paths_and_load_from_output(
        self, artifactorypath_mock, estimator_pickle_path_factory, open_estimator_pickle
    ):
        artifactorypath_mock.return_value.is_file.return_value = False

        paths = [
            estimator_pickle_path_factory(
                "LogisticRegression_2019_10_15_10_42_10_709197.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019_10_15_10_32_41_780990.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019_10_15_10_34_34_226695.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019_10_15_10_51_50_760746.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019_10_15_10_34_21_849358.pkl"
            ),
        ]

        mock_open = MagicMock()
        mock_open.open = open_estimator_pickle

        mock = MagicMock()
        mock.glob.return_value = paths
        mock.__truediv__.return_value = mock_open

        storage = ArtifactoryStorage("http://www.testy.com", "test", apikey="key")
        storage.artifactory_path = mock
        artifactory_paths = storage.get_list()

        estimator = storage.load(artifactory_paths[0])
        assert isinstance(estimator, (BaseEstimator, Pipeline))

        artifactorypath_mock.return_value.is_file.return_value = True

        estimator = storage.load(mock_open)
        assert isinstance(estimator, (BaseEstimator, Pipeline))

        assert artifactory_paths[0] == paths[1]
        assert artifactory_paths[-1] == paths[3]

    @require_artifactory
    def test_artifactory_initialization_path(self):
        from dohq_artifactory.auth import XJFrogArtApiAuth

        storage = ArtifactoryStorage("http://www.testy.com", "test", apikey="key")

        assert storage.artifactory_path.repo == "test"
        assert storage.artifactory_path.drive == "http://www.testy.com/artifactory"
        assert isinstance(storage.artifactory_path.auth, XJFrogArtApiAuth)
        assert storage.artifactory_path.auth.apikey == "key"

    @require_artifactory
    def test_artifactory_initialization_with_artifactory_suffix_works_as_expected(self):
        storage = ArtifactoryStorage("http://www.testy.com/artifactory", "test")
        assert storage.artifactory_path.repo == "test"
        assert storage.artifactory_path.drive == "http://www.testy.com/artifactory"
        assert storage.artifactory_path.auth is None

    @patch("ml_tooling.storage.artifactory._has_artifactory")
    def test_using_artifactory_without_installed_fails(self, mock_const):
        mock_const.__bool__.return_value = False
        with pytest.raises(
            MLToolingError,
            match="Artifactory not installed - run pip install dohq-artifactory",
        ):
            ArtifactoryStorage("test", "test")
