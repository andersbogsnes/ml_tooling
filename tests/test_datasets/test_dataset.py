import pandas as pd
import pytest

from ml_tooling.data import Dataset
from ml_tooling.utils import DatasetError


class TestDataset:
    def test_repr_is_correct(self, iris_dataset):
        result = str(iris_dataset())
        assert result == "<IrisData - Dataset>"

    def test_feature_names_access_works_correctly(self, iris_dataset):
        dataset = iris_dataset()
        assert dataset.features == [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

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
        self, california_csv, california_sqldataset, test_engine, california_filedataset
    ):
        csv_dataset = california_filedataset(california_csv)
        sql_dataset = california_sqldataset(test_engine, schema=None)

        csv_dataset.copy_to(sql_dataset)

        with sql_dataset.engine.connect() as conn:
            conn.execute("SELECT * FROM california")
