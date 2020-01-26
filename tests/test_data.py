import pandas as pd
import pytest

from ml_tooling.data import SQLDataset, Dataset, FileDataset
from ml_tooling.utils import DatasetError


class TestDataset:
    def test_repr_is_correct(self, IrisDataset):
        result = str(IrisDataset)
        assert result == "<IrisData - Dataset>"

    def test_dataset_x_attribute_access_works_correctly(self, IrisDataset):
        assert IrisDataset._x is None
        features = IrisDataset.x
        assert IrisDataset._x is not None
        assert len(features) == 150

    def test_dataset_y_attribute_access_works_correctly(self, IrisDataset):
        assert IrisDataset._y is None
        features = IrisDataset.y
        assert IrisDataset._y is not None
        assert len(features) == 150

    def test_dataset_repr_prints_correctly(self, IrisDataset):
        repr = str(IrisDataset)
        assert repr == "<IrisData - Dataset>"

    def test_cant_modify_x_and_y(self, IrisDataset):
        with pytest.raises(DatasetError, match="Trying to modify x - x is immutable"):
            IrisDataset.x = "testx"
        with pytest.raises(DatasetError, match="Trying to modify y - y is immutable"):
            IrisDataset.y = "testy"

    def test_dataset_has_validation_set_errors_correctly(self, IrisDataset):
        assert IrisDataset.has_validation_set is False
        IrisDataset.create_train_test()
        assert IrisDataset.has_validation_set is True

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

    def test_sqldataset_can_be_instantiated_with_engine_string(self):
        class BostonDataSet(SQLDataset):
            def load_training_data(self, *args, **kwargs):
                pass

            def load_prediction_data(self, *args, **kwargs):
                pass

        assert BostonDataSet("sqlite:///")

        with pytest.raises(ValueError, match="Invalid connection"):
            BostonDataSet(["not_a_url"])


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
