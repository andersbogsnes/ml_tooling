import pytest
import pathlib

from ml_tooling.storage import FileStorage

@pytest.fixture
def test_pickle_file(tmp_file: pathlib.Path, regression):
    storage = FileStorage(tmp_file)
    return regression.save_estimator(storage)