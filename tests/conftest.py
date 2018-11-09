import logging
import random as rand

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_tooling import BaseClassModel
from ml_tooling.transformers import DFStandardScaler

logging.disable(logging.CRITICAL)


@pytest.fixture(autouse=True)
def random():
    rand.seed(42)
    np.random.seed(42)


@pytest.fixture(name='base', scope='session')
def _base():
    class IrisModel(BaseClassModel):
        def get_prediction_data(self, idx):
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            return df.iloc[[idx]]

        def get_training_data(self):
            data = load_iris()
            y = np.where(data.target == 1, 1, 0)  # default roc_auc doesn't support multiclass
            x = pd.DataFrame(data.data, columns=data.feature_names)
            return x, y

    IrisModel.config.CROSS_VALIDATION = 2
    IrisModel.config.N_JOBS = 1
    return IrisModel


@pytest.fixture(name='categorical')
def categorical_data():
    return pd.DataFrame({"category_a": ["a1", "a2", "a3", "a1"],
                         "category_b": ["b1", "b2", "b3", "b1"]})


@pytest.fixture(name='categorical_na')
def categorical_na_data(categorical):
    categorical.loc[1, "category_a"] = np.nan
    categorical.loc[0, "category_b"] = np.nan
    return categorical


@pytest.fixture(name='numerical')
def numerical_data():
    return pd.DataFrame({"number_a": [1, 2, 3, 4],
                         "number_b": [5, 6, 7, 8]})


@pytest.fixture(name='numerical_na')
def _numerical_na_data(numerical):
    numerical.loc[0, "number_a"] = np.nan
    numerical.loc[3, "number_b"] = np.nan
    return numerical


@pytest.fixture(name='dates')
def dates_data():
    return pd.DataFrame({"date_a": pd.to_datetime(['2018-01-01',
                                                   '2018-02-01',
                                                   '2018-03-01'], format='%Y-%m-%d')})


@pytest.fixture(name='regression', scope='session')
def _linear_regression(base):
    model = base(LinearRegression())
    model.score_model()
    return model


@pytest.fixture(name='regression_cv', scope='session')
def _linear_regression_cv(base):
    model = base(LinearRegression())
    model.score_model(cv=2)
    return model


@pytest.fixture(name='classifier', scope='session')
def _logistic_regression(base):
    model = base(LogisticRegression(solver='liblinear'))
    model.score_model()
    return model


@pytest.fixture(name='classifier_cv', scope='session')
def _logistic_regression_cv(base):
    model = base(LogisticRegression(solver='liblinear'))
    model.score_model(cv=2)
    return model


@pytest.fixture
def pipeline_logistic(base):
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ])

    return pipe


@pytest.fixture
def pipeline_linear():
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LinearRegression())
    ])

    return pipe


@pytest.fixture
def pipeline_dummy_classifier():
    pipe = Pipeline([
        ('scale', DFStandardScaler()),
        ('clf', DummyClassifier())
    ])

    return pipe


@pytest.fixture
def pipeline_forest_classifier():
    pipe = Pipeline([
        ('scale', DFStandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=10))
    ])

    return pipe


@pytest.fixture
def monkeypatch_git_hash(monkeypatch):
    def mockreturn():
        return '1234'

    monkeypatch.setattr('ml_tooling.utils.get_git_hash', mockreturn)
