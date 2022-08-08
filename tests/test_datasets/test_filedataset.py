import pathlib

import pandas as pd
import pytest

from ml_tooling import Model
from ml_tooling.utils import DatasetError


class TestFileDataset:
    def test_filedataset_repr_prints_correctly(
        self, california_filedataset, california_csv
    ):
        result = repr(california_filedataset(california_csv))

        assert result == "<CaliforniaFileDataset - FileDataset>"

    def test_load_data_works_as_expected(
        self, california_filedataset, california_csv, tmp_path
    ):
        dataset = california_filedataset(california_csv)
        target = california_filedataset(tmp_path / "tmp.csv")
        assert not target.file_path.exists()
        target._load_data(dataset._dump_data())

        assert target.file_path.exists()

    def test_dump_data_works_as_expected(
        self, california_filedataset, california_csv, california_df
    ):
        dataset = california_filedataset(california_csv)

        result = dataset._dump_data()
        pd.testing.assert_frame_equal(result, california_df)

    def test_can_instantiate_filedataset(self, california_filedataset, california_csv):
        data = california_filedataset(california_csv)
        assert data.x is not None
        assert data.y is not None

    @pytest.mark.parametrize(
        "filename,extension", [("test.csv", "csv"), ("test.parquet", "parquet")]
    )
    def test_filedataset_extension_is_correct(
        self, filename, extension, california_filedataset
    ):
        dataset = california_filedataset(filename)
        assert dataset.extension == extension

    def test_filedataset_errors_if_given_folder(self, california_filedataset, tmp_path):
        assert tmp_path.suffix == ""
        with pytest.raises(DatasetError, match="must point to a file"):
            california_filedataset(tmp_path)

    def test_filedataset_that_returns_empty_training_data_raises_exception(
        self, california_csv, california_filedataset
    ):
        class FailingDataset(california_filedataset):
            def load_training_data(self, *args, **kwargs):
                return pd.DataFrame(dtype="object"), pd.Series(dtype="object")

        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_training_data"
        ):
            FailingDataset(california_csv).create_train_test()

    def test_filedataset_raises_exception_when_load_prediction_data_is_empty(
        self, regression: Model, california_filedataset, california_csv: pathlib.Path
    ):
        class FailingDataset(california_filedataset):
            def load_prediction_data(self, *args, **kwargs):
                return pd.DataFrame()

        data = FailingDataset(california_csv).create_train_test()

        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_prediction_data"
        ):
            regression.make_prediction(data, 0)
