import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def categorical():
    return pd.DataFrame(
        {"category_a": ["a1", "a2", "a3", "a1"], "category_b": ["b1", "b2", "b3", "b1"]}
    )


@pytest.fixture(scope="session")
def categorical_na(categorical):
    copy = categorical.copy()
    copy.loc[1, "category_a"] = np.nan
    copy.loc[0, "category_b"] = np.nan
    return copy


@pytest.fixture(scope="session")
def numerical():
    return pd.DataFrame({"number_a": [1, 2, 3, 4], "number_b": [5, 6, 7, 8]})


@pytest.fixture(scope="session")
def numerical_na(numerical):
    copy = numerical.copy()
    copy.loc[0, "number_a"] = np.nan
    copy.loc[3, "number_b"] = np.nan
    return copy


@pytest.fixture(scope="session")
def dates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_a": pd.to_datetime(
                ["2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01"],
                format="%Y-%m-%d",
            )
        }
    )
