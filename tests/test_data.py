import pytest
import sqlalchemy
from unittest.mock import MagicMock, Mock, patch
from sqlalchemy.exc import DBAPIError

from ml_tooling.data import SQLDataset, Dataset
from ml_tooling.utils import DatasetError


def test_repr_is_correct(test_dataset):
    result = str(test_dataset)
    assert result == "<IrisData - Dataset>"


def test_sqldataset_x_attribute_access_works_correctly(test_sqldata):
    assert test_sqldata._x is None
    features = test_sqldata.x
    assert test_sqldata._x is not None
    assert len(features) == 506


def test_sqldataset_y_attribute_access_works_correctly(test_sqldata):
    assert test_sqldata._y is None
    features = test_sqldata.y
    assert test_sqldata._y is not None
    assert len(features) == 506


def test_sqldataset_repr_prints_correctly(test_sqldata):
    repr = str(test_sqldata)
    assert repr == "<BostonData - SQLDataset Engine(sqlite:///:memory:)>"


def test_sqldataset_can_be_instantiated_with_engine_string(test_sqldata_class):
    assert test_sqldata_class("sqlite:///", "")

    with pytest.raises(ValueError, match="Invalid connection"):
        test_sqldata_class(["not_a_url"], "schema")


def test_sqldataset_errors_when_schema_is_defined_on_instantiation(
    test_sqldata_class, test_db
):
    test_sqldata_class.table = MagicMock()

    with pytest.raises(
        DatasetError,
        match="cannot have a defined schema - remove the schema declaration",
    ):
        assert test_sqldata_class(test_db, "schema")


def test_cant_modify_x_and_y(test_dataset):
    with pytest.raises(DatasetError, match="Trying to modify x - x is immutable"):
        test_dataset.x = "testx"
    with pytest.raises(DatasetError, match="Trying to modify y - y is immutable"):
        test_dataset.y = "testy"


def test_dataset_has_validation_set_errors_correctly(base_dataset):
    data = base_dataset()
    assert data.has_validation_set is False
    data.create_train_test()
    assert data.has_validation_set is True


def test_can_instantiate_filedataset(test_filedata, test_csv):
    data = test_filedata(test_csv)
    assert data.x is not None
    assert data.y is not None


def test_cannot_instantiate_an_abstract_baseclass():
    with pytest.raises(TypeError):
        Dataset()


def test_dataset_can_copy_to_table(
    test_sqldata, test_sqldata_class, test_db, test_table
):
    copy_to_data = test_sqldata_class(test_db, "main")
    copy_to_data.table = test_table

    test_sqldata.table = test_table
    test_sqldata.copy_to(target=copy_to_data)

    assert copy_to_data._load_prediction_data().index.all(
        [
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
        ]
    )


@patch("ml_tooling.data.sql.SQLDataset.create_connection")
def test_dump_data_throws_error_on_exec_failure(
    create_conn_mock, test_sqldata, test_table
):
    test_sqldata.table = test_table

    failing_conn_mock = MagicMock()
    failing_conn_mock.execute.side_effect = DBAPIError("test", "test", "test")
    create_conn_mock.return_value.__enter__.return_value = failing_conn_mock
    with pytest.raises(DBAPIError):
        test_sqldata._dump_data()


@patch("ml_tooling.data.sql.SQLDataset.create_connection")
def test_insert_data_throws_error_on_exec_failure(
    create_conn_mock, test_sqldata, test_table
):
    test_sqldata.table = test_table

    failing_conn_mock = MagicMock()
    failing_conn_mock.execute.side_effect = DBAPIError("test", "test", "test")
    trans_mock = MagicMock()

    create_conn_mock.return_value.__enter__.return_value = failing_conn_mock
    create_conn_mock.begin.return_value = trans_mock

    with pytest.raises(DBAPIError):
        test_sqldata._insert_data(MagicMock())
        trans_mock.rollback.assert_called()


@patch("sqlalchemy.schema.CreateSchema")
def test_setup_table_creates_schema_if_has_schema_returns_false(
    create_schema_mock, test_sqldata
):
    engine_mock = MagicMock()
    engine_mock.dialect.has_schema.return_value = False

    test_sqldata.engine = engine_mock
    test_sqldata.table = MagicMock()
    test_sqldata._setup_table(MagicMock())

    create_schema_mock.assert_called()
