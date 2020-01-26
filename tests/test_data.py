import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.exc import DBAPIError

from ml_tooling.data import Dataset, FileDataset, SQLDataset
from ml_tooling.utils import DatasetError


class TestDataset:
    def test_repr_is_correct(self, iris_dataset):
        result = str(iris_dataset)
        assert result == "<IrisData - Dataset>"

    def test_dataset_x_attribute_access_works_correctly(self, iris_dataset):
        assert iris_dataset._x is None
        features = iris_dataset.x
        assert iris_dataset._x is not None
        assert len(features) == 150

    def test_dataset_y_attribute_access_works_correctly(self, iris_dataset):
        assert iris_dataset._y is None
        features = iris_dataset.y
        assert iris_dataset._y is not None
        assert len(features) == 150

    def test_dataset_repr_prints_correctly(self, iris_dataset):
        repr = str(iris_dataset)
        assert repr == "<IrisData - Dataset>"

    def test_cant_modify_x_and_y(self, iris_dataset):
        with pytest.raises(DatasetError, match="Trying to modify x - x is immutable"):
            iris_dataset.x = "testx"
        with pytest.raises(DatasetError, match="Trying to modify y - y is immutable"):
            iris_dataset.y = "testy"

    def test_dataset_has_validation_set_errors_correctly(self, iris_dataset):
        assert iris_dataset.has_validation_set is False
        iris_dataset.create_train_test()
        assert iris_dataset.has_validation_set is True

    def test_cannot_instantiate_an_abstract_baseclass(self):
        with pytest.raises(TypeError):
            Dataset()

    def test_can_copy_one_dataset_into_another(
        self, boston_csv, boston_sqldataset, test_engine
    ):
        class EmptyDataset(FileDataset):
            def load_prediction_data(self, *args, **kwargs):
                pass

            def load_training_data(self, *args, **kwargs):
                pass

        class BostonData(SQLDataset):
            def load_training_data(self, *args, **kwargs):
                sql = "SELECT * FROM boston"
                df = pd.read_sql(sql, self.engine, index_col="index")
                return df.drop(columns="MEDV"), df.MEDV

            def load_prediction_data(self, *args, **kwargs):
                sql = "SELECT * FROM boston"
                df = pd.read_sql(sql, self.engine, index_col="index")
                return df.iloc[0]

        csv_dataset = EmptyDataset(boston_csv)
        sql_dataset = BostonData(test_engine)

        csv_dataset.copy_to(sql_dataset)

        with sql_dataset.engine.connect() as conn:
            conn.execute("SELECT * FROM boston")


class TestSqlDataset:
    def test_sqldataset_repr_prints_correctly(self, boston_sqldataset):
        repr = str(boston_sqldataset)
        assert repr == "<BostonData - SQLDataset Engine(sqlite:///:memory:)>"


def test_sqldataset_can_be_instantiated_with_engine_string(test_sqldata_class):
    assert test_sqldata_class("sqlite:///", "")

    def test_sqldataset_can_be_instantiated_with_engine_string(self):
        class BostonDataSet(SQLDataset):
            def load_training_data(self, *args, **kwargs):
                pass

            def load_prediction_data(self, *args, **kwargs):
                pass

        assert BostonDataSet("sqlite:///")

    with pytest.raises(ValueError, match="Invalid connection"):
        test_sqldata_class(["not_a_url"], "schema")


def test_sqldataset_errors_when_schema_is_defined_on_instantiation(
    test_sqldata_class, loaded_boston_db, boston_dataset
):
    test_sqldata_class.table = MagicMock()

    with pytest.raises(
        DatasetError,
        match="cannot have a defined schema - remove the schema declaration",
    ):
        test_sqldata_class(loaded_boston_db, "schema")
        with pytest.raises(ValueError, match="Invalid connection"):
            boston_dataset(["not_a_url"])


class TestFileDataset:
    def test_can_instantiate_filedataset(self, test_filedata, boston_csv):
        data = test_filedata(boston_csv)
        assert data.x is not None
        assert data.y is not None

    @pytest.mark.parametrize(
        "filename,extension", [("test.csv", "csv"), ("test.parquet", "parquet")]
    )
    def test_filedataset_extension_is_correct(self, filename, extension):
        class TestData(FileDataset):
            def load_prediction_data(self, *args, **kwargs):
                pass

            def load_training_data(self, *args, **kwargs):
                pass

        dataset = TestData(filename)
        assert dataset.extension == extension


def test_dataset_can_copy_to_table(
    test_sqldata, test_sqldata_class, loaded_boston_db, test_table
):
    copy_to_data = test_sqldata_class(loaded_boston_db, "main")
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
