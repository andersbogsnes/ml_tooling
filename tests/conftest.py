import matplotlib;

matplotlib.use('Agg')  # noqa
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from ml_utils import BaseClassModel
from sklearn.datasets import load_iris


@pytest.fixture(name='base', scope='session')
def _base():
    class IrisModel(BaseClassModel):
        def get_prediction_data(self, *args):
            x, _ = load_iris(return_X_y=True)
            idx = np.random.randint(len(x))
            return x[idx]

        def get_training_data(self):
            x, y = load_iris(return_X_y=True)
            y = np.where(y == 1, 1, 0)  # default roc_auc doesn't support multiclass
            return x, y

    return IrisModel


@pytest.fixture(name='categorical')
def categorical_data():
    return pd.DataFrame({"category_a": ["a1", "a2", "a3"],
                         "category_b": ["b1", "b2", "b3"]})


@pytest.fixture(name='categorical_na')
def categorical_na_data(categorical):
    categorical.loc[1, "category_a"] = np.nan
    categorical.loc[0, "category_b"] = np.nan
    return categorical


@pytest.fixture(name='numerical')
def numerical_data():
    return pd.DataFrame({"number_a": [1, 2, 3, 4],
                         "number_b": [5, 6, 7, 8]})


@pytest.fixture(name='dates')
def dates_data():
    return pd.DataFrame({"date_a": pd.to_datetime(['2018-01-01',
                                                   '2018-02-01',
                                                   '2018-03-01'], format='%Y-%m-%d')})


@pytest.fixture(name='regression', scope='session')
def _linear_regression(base):
    model = base(LinearRegression())
    model.set_config({"CROSS_VALIDATION": 2, "N_JOBS": 1})
    model.test_model()
    return model


@pytest.fixture(name='classifier', scope='session')
def _logistic_regression(base):
    model = base(LogisticRegression())
    model.set_config({"CROSS_VALIDATION": 2, "N_JOBS": 1})
    model.test_model()
    return model
