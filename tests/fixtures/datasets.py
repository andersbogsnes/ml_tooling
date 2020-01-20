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
def test_table():
    meta = sa.MetaData()
    return sa.Table(
        "boston",
        meta,
        sa.Column("CRIM", sa.Float),
        sa.Column("ZN", sa.Float),
        sa.Column("INDUS", sa.Float),
        sa.Column("CHAS", sa.Float),
        sa.Column("NOX", sa.Float),
        sa.Column("RM", sa.Float),
        sa.Column("AGE", sa.Float),
        sa.Column("DIS", sa.Float),
        sa.Column("RAD", sa.Float),
        sa.Column("TAX", sa.Float),
        sa.Column("PTRATIO", sa.Float),
        sa.Column("B", sa.Float),
        sa.Column("LSTAT", sa.Float),
    )


@pytest.fixture
def test_engine():
    engine = sa.create_engine("sqlite:///:memory:")
    engine.dialect.has_schema = lambda *args: True
    return engine


@pytest.fixture
def test_db(test_df, test_engine):
    test_df.to_sql("boston", test_engine)
    return test_engine


@pytest.fixture
def test_sqldata_class():
    class BostonData(SQLDataset):
        def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
            sql = f"SELECT * FROM boston"
            df = pd.read_sql(sql, self.engine)
            return df.drop(columns="MEDV"), df.MEDV

        def load_prediction_data(self, *args, **kwargs) -> DataType:
            sql = f"SELECT * FROM boston"
            df = pd.read_sql(sql, self.engine)
            return df.iloc[0]

    return BostonData


@pytest.fixture
def test_sqldata(test_sqldata_class, test_db):
    return test_sqldata_class(test_db, "")


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
