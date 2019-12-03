import pathlib
from unittest.mock import patch, MagicMock

import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.utils import MLToolingError
from ml_tooling.storage import Storage
from ml_tooling.storage.file import FileStorage


class TestFileStorage:
    def test_can_save_file(self, classifier: Model, tmp_path: pathlib.Path):
        storage = FileStorage(tmp_path)
        expected_file = storage.save(classifier.estimator, "estimator")
        assert expected_file.exists()

    def test_can_save_with_model(self, classifier: Model, tmp_path: pathlib.Path):
        storage = FileStorage(tmp_path)
        expected_file = classifier.save_estimator(storage)
        assert expected_file.exists()

        storage_context = FileStorage(tmp_path)
        context_expected_file = classifier.save_estimator(storage_context)
        assert context_expected_file.exists()

    def test_can_load_file(self, classifier: Model, tmp_path: pathlib.Path):
        storage = FileStorage(tmp_path)
        storage.save(classifier.estimator, "estimator")
        loaded_file = storage.load("estimator")
        assert isinstance(loaded_file, (BaseEstimator, Pipeline))

    def test_can_load_file_by_filename(self, classifier: Model, tmp_path: pathlib.Path):
        storage = FileStorage(tmp_path)
        storage.save(classifier.estimator, "estimator.file")
        loaded_file = storage.load("estimator.file")
        assert isinstance(loaded_file, (BaseEstimator, Pipeline))
        assert classifier.estimator.get_params() == loaded_file.get_params()

    def test_can_load_with_model(self, classifier: Model, tmp_path: pathlib.Path):
        storage = FileStorage(tmp_path)
        expected_file = classifier.save_estimator(storage)
        assert expected_file.exists()
        loaded_file = classifier.load_estimator(storage, expected_file)
        assert isinstance(loaded_file, Model)
        storage_context = FileStorage(tmp_path)
        context_loaded_file = classifier.load_estimator(storage_context, expected_file)
        assert isinstance(context_loaded_file, Model)

    def test_can_list_estimators(self, classifier: Model, tmp_path: pathlib.Path):
        storage = FileStorage(tmp_path)
        for _ in range(3):
            classifier.save_estimator(storage)
        storage_context = FileStorage(tmp_path)
        filenames_list = Model.list_estimators(storage_context)
        for filename in filenames_list:
            assert filename.exists()

    def test_can_get_list_of_paths_and_load_from_output(
        self, estimator_pickle_path_factory, tmp_path
    ):
        paths = [
            estimator_pickle_path_factory(
                "LogisticRegression_2019-10-15_10:42:10.709197.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019-10-15_10:32:41.780990.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019-10-15_10:34:34.226695.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019-10-15_10:51:50.760746.pkl"
            ),
            estimator_pickle_path_factory(
                "LogisticRegression_2019-10-15_10:34:21.849358.pkl"
            ),
        ]

        storage = FileStorage(tmp_path)

        estimators = storage.get_list()

        first_estimator = storage.load(estimators[0])

        assert isinstance(first_estimator, (BaseEstimator, Pipeline))
        assert estimators[0] == paths[1]
        assert estimators[-1] == paths[3]

    def test_raise_when_non_dir(self, classifier: Model, tmp_path: pathlib.Path):
        with pytest.raises(
            MLToolingError, match="dir_path is /not/a/dir.file which is not a directory"
        ):
            _ = FileStorage("/not/a/dir.file")

    def test_cannot_instantiate_an_abstract_baseclass(self):
        with pytest.raises(TypeError):
            Storage()

    @patch("ml_tooling.storage.file.find_src_dir")
    def test_store_prod_flag_overrules_init_(
        self, mock_dir: MagicMock, tmp_path: pathlib.Path, classifier: Model
    ):
        other_folder = tmp_path / "someotherfolder"
        other_folder.mkdir()
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        mock_dir.return_value = src_dir

        storage = FileStorage(other_folder)
        storage.save(classifier.estimator, "prod.pkl", prod=True)
        assert src_dir.joinpath("prod.pkl").exists()

        mock_dir.assert_called_with()
