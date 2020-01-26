import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.exc import DBAPIError

from ml_tooling.data import Dataset
from ml_tooling.utils import DatasetError


class TestDataset:
    def test_repr_is_correct(self, iris_dataset):
        result = str(iris_dataset())
        assert result == "<IrisData - Dataset>"

    def test_dataset_x_attribute_access_works_correctly(self, iris_dataset):
        dataset = iris_dataset()
        assert dataset._x is None
        features = dataset.x
        assert dataset._x is not None
        assert len(features) == 150

    def test_dataset_y_attribute_access_works_correctly(self, iris_dataset):
        dataset = iris_dataset()
        assert dataset._y is None
        features = dataset.y
        assert dataset._y is not None
        assert len(features) == 150

    def test_cant_modify_x_and_y(self, iris_dataset):
        dataset = iris_dataset()
        with pytest.raises(DatasetError, match="Trying to modify x - x is immutable"):
            dataset.x = "testx"
        with pytest.raises(DatasetError, match="Trying to modify y - y is immutable"):
            dataset.y = "testy"

    def test_dataset_has_validation_set_errors_correctly(self, iris_dataset):
        dataset = iris_dataset()
        assert dataset.has_validation_set is False
        dataset.create_train_test()
        assert dataset.has_validation_set is True

    def test_cannot_instantiate_an_abstract_baseclass(self):
        with pytest.raises(TypeError):
            Dataset()

    def test_can_copy_one_dataset_into_another(
        self, boston_csv, boston_sqldataset, test_engine, boston_filedataset
    ):
        csv_dataset = boston_filedataset(boston_csv)
        sql_dataset = boston_sqldataset(test_engine, schema=None)

        csv_dataset.copy_to(sql_dataset)

        with sql_dataset.engine.connect() as conn:
            conn.execute("SELECT * FROM boston")


class TestSqlDataset:
    def test_sqldataset_repr_prints_correctly(self, boston_sqldataset, test_engine):
        repr = str(boston_sqldataset(test_engine, schema=""))
        assert repr == "<BostonSQLData - SQLDataset Engine(sqlite:///:memory:)>"

    def test_sqldataset_can_be_instantiated_with_engine_string(self, iris_sqldataset):
        assert iris_sqldataset("sqlite:///", "")
        with pytest.raises(ValueError, match="Invalid connection"):
            iris_sqldataset(["not_a_url"], "schema")

    def test_sqldataset_errors_when_schema_is_defined_on_instantiation(
        self, boston_sqldataset, test_engine
    ):
        schema = MagicMock()
        schema.return_value = "something"
        boston_sqldataset.table.schema = schema

        with pytest.raises(
            DatasetError,
            match="cannot have a defined schema - remove the schema declaration",
        ):
            boston_sqldataset(test_engine, "schema")

    def test_sqldataset_can_copy_to_another_sqldataset(
        self, boston_sqldataset, loaded_boston_db
    ):
        source_data = boston_sqldataset(loaded_boston_db, "main")
        target_data = boston_sqldataset(loaded_boston_db, schema=None)

        source_data.copy_to(target=target_data)

        assert set(target_data._load_prediction_data(idx=0).columns) == {
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
        }

    @patch("ml_tooling.data.sql.pd.read_sql")
    def test_dump_data_throws_error_on_exec_failure(
        self, read_sql, boston_sqldataset, test_engine
    ):
        read_sql.side_effect = DBAPIError("test", "test", "test")

        with pytest.raises(DBAPIError):
            boston_sqldataset(test_engine, "")._dump_data()
        read_sql.assert_called_once()

    def test_load_data_throws_error_on_exec_failure(
        self, boston_sqldataset, test_engine
    ):
        dataset = boston_sqldataset(test_engine, None)
        create_conn_mock = MagicMock()
        failing_conn_mock = MagicMock()
        failing_conn_mock.execute.side_effect = DBAPIError("test", "test", "test")
        trans_mock = MagicMock()
        failing_conn_mock.begin.return_value = trans_mock

        create_conn_mock.return_value.__enter__.return_value = failing_conn_mock

        dataset.create_connection = create_conn_mock
        with pytest.raises(DBAPIError):
            dataset._load_data(MagicMock())
        trans_mock.rollback.assert_called()

    @patch("sqlalchemy.schema.CreateSchema")
    def test_setup_table_creates_schema_if_has_schema_returns_false(
        self, create_schema_mock, boston_sqldataset, test_engine
    ):
        dataset = boston_sqldataset(test_engine, "")

        engine_mock = MagicMock()
        engine_mock.dialect.has_schema.return_value = False
        dataset.engine = engine_mock

        dataset._setup_table(MagicMock())

        create_schema_mock.assert_called()


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
