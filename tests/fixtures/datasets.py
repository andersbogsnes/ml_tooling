import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sklearn.datasets import load_iris, fetch_california_housing

from ml_tooling.data import SQLDataset, FileDataset
from ml_tooling.data.base_data import Dataset
from ml_tooling.utils import DataType


@pytest.fixture(scope="session")
def california_df():
    california_data = fetch_california_housing()
    return pd.DataFrame(
        data=california_data.data, columns=california_data.feature_names
    ).assign(MedHouseVal=california_data.target)


@pytest.fixture(scope="session")
def iris_df():
    iris_data = load_iris()
    y = np.where(iris_data.target == 1, 1, 0)
    return pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names).assign(
        target=y
    )


@pytest.fixture(scope="session")
def iris_dataset(iris_df):
    class IrisData(Dataset):
        def load_prediction_data(self, idx):
            return iris_df.drop(columns="target").iloc[[idx]]

        def load_training_data(self):
            return iris_df.drop(columns="target"), iris_df.target

    return IrisData


@pytest.fixture(scope="session")
def train_iris_dataset(iris_dataset):
    return iris_dataset().create_train_test(stratify=True)


@pytest.fixture(scope="session")
def california_dataset(california_df):
    class CaliforniaData(Dataset):
        def load_prediction_data(self, idx) -> pd.DataFrame:
            return california_df.iloc[[idx]]

        def load_training_data(self) -> Tuple[pd.DataFrame, DataType]:
            return california_df.drop(columns="MedHouseVal"), california_df.MedHouseVal

    return CaliforniaData


@pytest.fixture()
def train_california_dataset(california_dataset):
    return california_dataset().create_train_test(stratify=False)


@pytest.fixture
def test_engine():
    engine = sa.create_engine("sqlite:///:memory:")
    return engine


@pytest.fixture
def loaded_california_db(california_df, test_engine):
    california_df.to_sql("california", test_engine, index=False)
    return test_engine


@pytest.fixture
def loaded_iris_db(iris_df, test_engine):
    iris_df.to_sql("iris", test_engine)
    return test_engine


@pytest.fixture
def california_sqldataset():
    meta = sa.MetaData()

    class CaliforniaSQLData(SQLDataset):
        table = sa.Table(
            "california",
            meta,
            sa.Column("MedInc", sa.Float),
            sa.Column("HouseAge", sa.Float),
            sa.Column("AveRooms", sa.Float),
            sa.Column("AveBedrms", sa.Float),
            sa.Column("Population", sa.Float),
            sa.Column("AveOccup", sa.Float),
            sa.Column("Latitude", sa.Float),
            sa.Column("Longitude", sa.Float),
            sa.Column("MedHouseVal", sa.Float),
        )

        def load_training_data(
            self, conn, *args, **kwargs
        ) -> Tuple[pd.DataFrame, DataType]:
            sql = "SELECT * FROM california"
            df = pd.read_sql(sql, conn)
            return df.drop(columns="MedHouseVal"), df.MedHouseVal

        def load_prediction_data(self, idx, conn) -> DataType:
            sql = "SELECT * FROM california"
            return pd.read_sql(sql, conn).loc[[idx]].drop(columns="MedHouseVal")

    return CaliforniaSQLData


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
def california_csv(tmp_path: pathlib.Path, california_df):
    output_path = tmp_path / "test.csv"
    california_df.to_csv(output_path, index=False)
    return output_path


@pytest.fixture()
def california_filedataset():
    class CaliforniaFileDataset(FileDataset):
        def load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
            df = pd.read_csv(self.file_path)
            return df.drop(columns=["MedHouseVal"]), df.MedHouseVal

        def load_prediction_data(self, idx) -> pd.DataFrame:
            df = pd.read_csv(self.file_path)
            return df.drop(columns="MedHouseVal").iloc[[idx]]

    return CaliforniaFileDataset


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
