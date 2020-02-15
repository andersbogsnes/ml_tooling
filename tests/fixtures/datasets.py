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
def boston_df():
    boston_data = load_boston()
    return pd.DataFrame(
        data=boston_data.data, columns=boston_data.feature_names
    ).assign(MEDV=boston_data.target)


@pytest.fixture
def iris_df():
    iris_data = load_iris()
    y = np.where(iris_data.target == 1, 1, 0)
    return pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names).assign(
        target=y
    )


@pytest.fixture
def iris_dataset(iris_df):
    class IrisData(Dataset):
        def load_prediction_data(self, idx):
            return iris_df.drop(columns="target").iloc[[idx]]

        def load_training_data(self):
            return iris_df.drop(columns="target"), iris_df.target

    return IrisData


@pytest.fixture()
def train_iris_dataset(iris_dataset):
    return iris_dataset().create_train_test(stratify=True)


@pytest.fixture
def boston_dataset(boston_df):
    class BostonData(Dataset):
        def load_prediction_data(self, idx) -> pd.DataFrame:
            return boston_df.iloc[[idx]]

        def load_training_data(self) -> Tuple[pd.DataFrame, DataType]:
            return boston_df.drop(columns="MEDV"), boston_df.MEDV

    return BostonData


@pytest.fixture()
def train_boston_dataset(boston_dataset):
    return boston_dataset().create_train_test(stratify=False)


@pytest.fixture
def test_engine():
    engine = sa.create_engine("sqlite:///:memory:")
    engine.dialect.has_schema = lambda *args: True
    return engine


@pytest.fixture
def loaded_boston_db(boston_df, test_engine):
    boston_df.to_sql("boston", test_engine, index=False)
    return test_engine


@pytest.fixture
def loaded_iris_db(iris_df, test_engine):
    iris_df.to_sql("iris", test_engine)
    return test_engine


@pytest.fixture
def boston_sqldataset():
    meta = sa.MetaData()

    class BostonSQLData(SQLDataset):
        table = sa.Table(
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
            sa.Column("MEDV", sa.Float),
        )

        def load_training_data(
            self, conn, *args, **kwargs
        ) -> Tuple[pd.DataFrame, DataType]:
            sql = "SELECT * FROM boston"
            df = pd.read_sql(sql, conn)
            return df.drop(columns="MEDV"), df.MEDV

        def load_prediction_data(self, idx, conn) -> DataType:
            sql = "SELECT * FROM boston"
            return pd.read_sql(sql, conn).loc[[idx]].drop(columns="MEDV")

    return BostonSQLData


@pytest.fixture
def iris_sqldataset():
    meta = sa.MetaData()

    class IrisSQLData(SQLDataset):
        table = sa.Table(
            "iris",
            meta,
            sa.Column("sepal length (cm)", sa.Float),
            sa.Column("sepal width (cm)", sa.Float),
            sa.Column("petal length (cm)", sa.Float),
            sa.Column("petal width (cm)", sa.Float),
        )

        def load_training_data(
            self, conn, *args, **kwargs
        ) -> Tuple[pd.DataFrame, DataType]:
            sql = "SELECT * FROM iris"
            df = pd.read_sql(sql, self.engine)
            return df.drop(columns="target"), df.target

        def load_prediction_data(self, idx, conn) -> pd.DataFrame:
            sql = "SELECT * FROM iris"
            return pd.read_sql(sql, self.engine).loc[[idx]].drop(columns="target")

    return IrisSQLData


@pytest.fixture
def boston_csv(tmp_path: pathlib.Path, boston_df):
    output_path = tmp_path / "test.csv"
    boston_df.to_csv(output_path, index=False)
    return output_path


@pytest.fixture()
def boston_filedataset():
    class BostonFileDataset(FileDataset):
        def load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
            df = pd.read_csv(self.file_path)
            return df.drop(columns=["MEDV"]), df.MEDV

        def load_prediction_data(self, idx) -> pd.DataFrame:
            df = pd.read_csv(self.file_path)
            return df.drop(columns="MEDV").iloc[[idx]]

    return BostonFileDataset


@pytest.fixture()
def iris_filedataset():
    class IrisFileDataset(FileDataset):
        def load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
            df = pd.read_csv(self.file_path)
            return df.drop(columns="target"), df.target

        def load_prediction_data(self, idx) -> pd.DataFrame:
            df = pd.read_csv(self.file_path)
            return df.drop(columns="target").iloc[[idx]]

    return IrisFileDataset
