import pathlib

import pandas as pd
import pytest

from ml_tooling import Model
from ml_tooling.utils import DatasetError


class TestFileDataset:
    def test_filedataset_repr_prints_correctly(self, boston_filedataset, boston_csv):
        result = repr(boston_filedataset(boston_csv))

        assert result == "<BostonFileDataset - FileDataset>"

    def test_load_data_works_as_expected(
        self, boston_filedataset, boston_csv, tmp_path
    ):
        dataset = boston_filedataset(boston_csv)
        target = boston_filedataset(tmp_path / "tmp.csv")
        assert not target.file_path.exists()
        target._load_data(dataset._dump_data())

        assert target.file_path.exists()

    def test_dump_data_works_as_expected(
        self, boston_filedataset, boston_csv, boston_df
    ):
        dataset = boston_filedataset(boston_csv)

        result = dataset._dump_data()
        pd.testing.assert_frame_equal(result, boston_df)

    def test_can_instantiate_filedataset(self, boston_filedataset, boston_csv):
        data = boston_filedataset(boston_csv)
        assert data.x is not None
        assert data.y is not None

    @pytest.mark.parametrize(
        "filename,extension", [("test.csv", "csv"), ("test.parquet", "parquet")]
    )
    def test_filedataset_extension_is_correct(
        self, filename, extension, boston_filedataset
    ):
        dataset = boston_filedataset(filename)
        assert dataset.extension == extension

    def test_filedataset_errors_if_given_folder(self, boston_filedataset, tmp_path):
        assert tmp_path.suffix == ""
        with pytest.raises(DatasetError, match="must point to a file"):
            boston_filedataset(tmp_path)

    def test_filedataset_that_returns_empty_training_data_raises_exception(
        self, boston_csv, boston_filedataset
    ):
        class FailingDataset(boston_filedataset):
            def load_training_data(self, *args, **kwargs):
                return pd.DataFrame(dtype="object"), pd.Series(dtype="object")

        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_training_data"
        ):
            FailingDataset(boston_csv).create_train_test()

    def test_filedataset_raises_exception_when_load_prediction_data_is_empty(
        self, regression: Model, boston_filedataset, boston_csv: pathlib.Path
    ):
        class FailingDataset(boston_filedataset):
            def load_prediction_data(self, *args, **kwargs):
                return pd.DataFrame()

        data = FailingDataset(boston_csv).create_train_test()

        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_prediction_data"
        ):
            regression.make_prediction(data, 0)
