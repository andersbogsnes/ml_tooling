import pytest
from ml_tooling.utils import DataSetError


def test_sqldataset_works_correctly(test_sqldata):
    assert test_sqldata._x is None
    features = test_sqldata.x
    assert test_sqldata._x is not None
    assert len(features) == 506


def test_cant_modify_x_and_y(test_dataset):
    with pytest.raises(DataSetError, match="Trying to modify x - x is immutable"):
        test_dataset.x = "testx"
    with pytest.raises(DataSetError, match="Trying to modify y - y is immutable"):
        test_dataset.y = "testy"


def test_dataset_has_validation_set_errors_correctly(base_dataset):

    data = base_dataset()
    assert data.has_validation_set is False
    data.create_train_test()
    assert data.has_validation_set is True
