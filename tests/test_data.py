from typing import Tuple

import pytest

from sklearn.datasets import load_boston
import pandas as pd
import sqlalchemy as sa
from ml_tooling.data.sql import SQLDataSet
from ml_tooling.utils import DataType


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
def test_data(test_db):
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


def test_tabledataset_works_correctly(test_data):
    assert test_data._x is None
    features = test_data.x
    assert test_data._x is not None
    assert len(features) == 506
