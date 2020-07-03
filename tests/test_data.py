import pathlib
from typing import Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from sklearn.datasets import load_iris
from sqlalchemy.exc import DBAPIError

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.data.demo_dataset import load_demo_dataset
from ml_tooling.utils import DatasetError, DataType

from sklearn.linear_model import LogisticRegression


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
        dataset.create_train_test(stratify=True)
        assert dataset.has_validation_set is True

    def test_dataset_that_returns_empty_training_data_errors_correctly(self):
        class FailingDataset(Dataset):
            def load_training_data(self, *args, **kwargs):
                return pd.DataFrame(), None

            def load_prediction_data(self, *args, **kwargs):
                pass

        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_training_data"
        ):
            FailingDataset()._load_training_data()

    def test_dataset_raises_when_load_prediction_data_returns_empty(self, classifier):
        class FailingDataset(Dataset):
            def load_training_data(self, *args, **kwargs):
                pass

            def load_prediction_data(self, *args, **kwargs):
                return pd.DataFrame()

        data = FailingDataset()
        with pytest.raises(
            DatasetError, match="An empty dataset was returned by load_prediction_data"
        ):
            classifier.make_prediction(data, 0)

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
                return pd.DataFrame(), pd.Series()

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
                return pd.DataFrame(), pd.Series()

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


class TestDemoDatasetModule:
    @pytest.fixture
    def load_dataset_iris(self) -> Dataset:
        return load_demo_dataset("iris")

    @pytest.fixture
    def iris_df(self):
        iris_data = load_iris()
        return (
            pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names),
            iris_data.target,
        )

    def test_repr_is_correct_load(self, load_dataset_iris: Dataset):
        result = str(load_dataset_iris)
        assert result == "<DemoData - Dataset>"

    def test_dataset_return_correct_x_attribute(
        self, load_dataset_iris: Dataset, iris_df: Tuple[pd.DataFrame, DataType]
    ):
        x_expected, y_expected = iris_df
        pd.testing.assert_frame_equal(load_dataset_iris.x, x_expected)

    def test_dataset_return_correct_y_attribute(
        self, load_dataset_iris: Dataset, iris_df: Tuple[pd.DataFrame, DataType]
    ):
        x_expected, y_expected = iris_df
        assert np.array_equal(load_dataset_iris.y, y_expected)

    def test_dataset_from_fetchopenml_works(self):
        dataset = load_demo_dataset("openml", name="miceprotein")
        assert len(dataset.x) == 1080

    def test_dataset_x_from_fetchopenml_with_paramteres_works(self):
        dataset = load_demo_dataset(
            "openml", name="blood-transfusion-service-center", target_column="V1"
        )
        features_x = dataset.x
        assert features_x.shape == (748, 4)

    def test_dataset_y_from_fetchopenml_with_two_target_columns_works(self):
        dataset = load_demo_dataset(
            "openml",
            name="blood-transfusion-service-center",
            target_column=["V1", "V2"],
        )
        features_y = dataset.y
        assert features_y.shape == (748, 2)

    def test_load_prediction_data_works_as_expected(self):
        dataset = load_demo_dataset("iris")
        dataset.create_train_test(stratify=True)
        model = Model(LogisticRegression())
        model.train_estimator(dataset)
        result = model.make_prediction(dataset, 5)

        expected = pd.DataFrame({"Prediction": [0]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
