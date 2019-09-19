from typing import Tuple

import pytest

from sklearn.datasets import load_boston
import pandas as pd
import sqlalchemy as sa
from ml_tooling.data.sql import SQLDataSet
from ml_tooling.data.file import FileDataSet
from ml_tooling.utils import DataType, DataSetError


@pytest.fixture
def test_df():
    boston_data = load_boston()
    return pd.DataFrame(
        data=boston_data.data, columns=boston_data.feature_names
    ).assign(MEDV=boston_data.target)


@pytest.fixture
def test_engine():
    return sa.create_engine("sqlite:///:memory:")


@pytest.fixture
def test_db(test_df, test_engine):
    test_df.to_sql("boston", test_engine)
    return test_engine


@pytest.fixture
def test_sqldata(test_db):
    class BostonDataSet(SQLDataSet):
        def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
            sql = "SELECT * FROM boston"
            df = pd.read_sql(sql, self.engine, index_col="index")
            return df.drop(columns="MEDV"), df.MEDV

        def load_prediction_data(self, *args, **kwargs) -> DataType:
            sql = "SELECT * FROM boston"
            df = pd.read_sql(sql, self.engine, index_col="index")
            return df.iloc[0]

    return BostonDataSet(test_db)


def test_sqldataset_works_correctly(test_sqldata):
    assert test_sqldata._x is None
    features = test_sqldata.x
    assert test_sqldata._x is not None
    assert len(features) == 506


def test_cant_modify_x_and_y(test_data):
    with pytest.raises(DataSetError, match="Trying to modify x - x is immutable"):
        test_data.x = "testx"
    with pytest.raises(DataSetError, match="Trying to modify y - y is immutable"):
        test_data.y = "testy"


def test_can_save_to_another_dataset(test_data, tmp_path):
    test_file = tmp_path / "test.parquet"
    filedata = FileDataSet(test_file)
    filedata.save()

    result = pd.read_parquet(test_file)
    assert result.drop(columns="MEDV") == test_data.x
