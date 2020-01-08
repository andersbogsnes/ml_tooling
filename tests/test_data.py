import pytest

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


def test_sqldataset_can_be_instantiated_with_engine_string():
    class BostonDataSet(SQLDataset):
        def load_training_data(self, *args, **kwargs):
            pass

        def load_prediction_data(self, *args, **kwargs):
            pass

    assert BostonDataSet("sqlite:///")

    with pytest.raises(ValueError, match="Invalid connection"):
        BostonDataSet(["not_a_url"])


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
