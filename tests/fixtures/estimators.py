import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_tooling.transformers import DFStandardScaler


@pytest.fixture()
def regression(base, test_dataset):
    model = base(LinearRegression())
    model.score_estimator(test_dataset)
    return model


@pytest.fixture()
def regression_cv(base, test_dataset):
    model = base(LinearRegression())
    model.score_estimator(test_dataset, cv=2)
    return model


@pytest.fixture()
def classifier(base, test_dataset):
    model = base(LogisticRegression(solver="liblinear"))
    model.score_estimator(test_dataset)
    return model


@pytest.fixture()
def classifier_cv(base, test_dataset):
    model = base(LogisticRegression(solver="liblinear"))
    model.score_estimator(test_dataset, cv=2)
    return model


@pytest.fixture
def pipeline_logistic(base):
    pipe = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear"))]
    )
    return pipe


@pytest.fixture
def pipeline_linear():
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LinearRegression())])
    return pipe


@pytest.fixture
def pipeline_dummy_classifier():
    pipe = Pipeline([("scale", DFStandardScaler()), ("clf", DummyClassifier())])
    return pipe


@pytest.fixture
def pipeline_forest_classifier():
    pipe = Pipeline(
        [
            ("scale", DFStandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10)),
        ]
    )
    return pipe
