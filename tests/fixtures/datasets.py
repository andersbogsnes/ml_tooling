import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sklearn.datasets import load_boston, load_iris

from ml_tooling.data import SQLDataset, FileDataset
from ml_tooling.data.base_data import Dataset
from ml_tooling.utils import DataType


@pytest.fixture
def base_dataset():
    class IrisData(Dataset):
        def load_prediction_data(self, idx):
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            return df.iloc[[idx]]

        def load_training_data(self):
            data = load_iris()
            y = np.where(
                data.target == 1, 1, 0
            )  # default roc_auc doesn't support multiclass
            x = pd.DataFrame(data.data, columns=data.feature_names)
            return x, y

    return IrisData


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
    class BostonData(SQLDataset):
        def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
            sql = "SELECT * FROM boston"
            df = pd.read_sql(sql, self.engine, index_col="index")
            return df.drop(columns="MEDV"), df.MEDV

        def load_prediction_data(self, *args, **kwargs) -> DataType:
            sql = "SELECT * FROM boston"
            df = pd.read_sql(sql, self.engine, index_col="index")
            return df.iloc[0]

    return BostonData(test_db)


@pytest.fixture
def test_csv(tmp_path: pathlib.Path):
    output_path = tmp_path / "test.csv"
    data = load_iris()
    y = np.where(data.target == 1, 1, 0)  # default roc_auc doesn't support multiclass
    df = pd.DataFrame(data.data, columns=data.feature_names).assign(target=y)
    df.to_csv(output_path)
    return output_path


@pytest.fixture
def test_filedata():
    class CSVData(FileDataset):
        def load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
            df = pd.read_csv(self.file_path)
            return df.drop(columns=["target"]), df.target

        def load_prediction_data(self, *args, **kwargs) -> pd.DataFrame:
            pass

    return CSVData


@pytest.fixture()
def test_dataset(base_dataset):
    return base_dataset().create_train_test()
