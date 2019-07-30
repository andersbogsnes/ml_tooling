import pytest

from sklearn.datasets import load_boston
import pandas as pd
import sqlalchemy as sa


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
    pass


def test_tabledataset_works_correctly(test_engine):
    pass
