from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.exc import DBAPIError

from ml_tooling.utils import DatasetError


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

    def test_sqldataset_can_load_training_data(
        self, boston_sqldataset, loaded_boston_db, boston_df
    ):
        dataset = boston_sqldataset(loaded_boston_db, None)
        x, y = dataset._load_training_data()
        pd.testing.assert_frame_equal(x.assign(MEDV=y), boston_df)

    def test_sqldataset_can_load_prediction_data(
        self, boston_sqldataset, boston_df, loaded_boston_db
    ):
        dataset = boston_sqldataset(loaded_boston_db, schema=None)
        result = dataset.load_prediction_data(0, conn=loaded_boston_db)

        expected = boston_df.iloc[[0], :].drop(columns="MEDV")

        pd.testing.assert_frame_equal(result, expected)

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

    def test_sql_dataset_raises_exception_when_load_training_data_returns_empty(
        self, boston_sqldataset
    ):
        class FailingDataset(boston_sqldataset):
            def load_training_data(self, *args, **kwargs):
                return pd.DataFrame(dtype="object"), pd.Series(dtype="object")

        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_training_data"
        ):
            FailingDataset("sqlite:///", schema=None).create_train_test()

    def test_sql_dataset_raises_exception_when_load_prediction_data_returns_empty(
        self, boston_sqldataset, regression
    ):
        class FailingDataset(boston_sqldataset):
            def load_prediction_data(self, *args, **kwargs):
                return pd.DataFrame()

        data = FailingDataset("sqlite:///", schema=None)
        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_prediction_data"
        ):
            regression.make_prediction(data, 0)

    def test_load_data_throws_error_on_exec_failure(
        self, boston_sqldataset, test_engine
    ):
        dataset = boston_sqldataset(test_engine, None)
        trans_mock = MagicMock()
        conn_mock = MagicMock()
        conn_mock.begin.return_value = trans_mock

        create_conn_mock = MagicMock()
        create_conn_mock.return_value.__enter__.return_value = conn_mock
        dataset.create_connection = create_conn_mock

        with pytest.raises(DBAPIError):
            data_mock = MagicMock()
            data_mock.to_sql.side_effect = DBAPIError("test", "test", "test")
            dataset._load_data(data_mock)
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
