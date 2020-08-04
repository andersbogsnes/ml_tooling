from typing import Callable, Optional

import pytest
import joblib
import pathlib

from _pytest.fixtures import FixtureRequest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml_tooling import Model
from ml_tooling.transformers import DFStandardScaler, DFFeatureUnion, Select


@pytest.fixture(scope="session")
def regression(train_iris_dataset) -> Model:
    model: Model = Model(LinearRegression())
    model.score_estimator(train_iris_dataset)
    return model


@pytest.fixture(scope="session")
def regression_cv(train_iris_dataset) -> Model:
    model: Model = Model(LinearRegression())
    model.score_estimator(train_iris_dataset, cv=2)
    return model


@pytest.fixture(scope="session")
def classifier(train_iris_dataset) -> Model:
    model: Model = Model(LogisticRegression(solver="liblinear"))
    model.score_estimator(train_iris_dataset)
    return model


@pytest.fixture(scope="session")
def classifier_cv(train_iris_dataset) -> Model:
    model: Model = Model(LogisticRegression(solver="liblinear"))
    model.score_estimator(train_iris_dataset, cv=2)
    return model


@pytest.fixture(scope="session")
def pipeline_logistic() -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("estimator", LogisticRegression(solver="liblinear")),
        ]
    )
    return pipe


@pytest.fixture(scope="session")
def pipeline_linear() -> Pipeline:
    pipe = Pipeline([("scale", StandardScaler()), ("estimator", LinearRegression())])
    return pipe


@pytest.fixture(scope="session")
def pipeline_dummy_classifier() -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", DFStandardScaler()),
            ("estimator", DummyClassifier(strategy="prior")),
        ]
    )
    return pipe


@pytest.fixture(scope="session")
def feature_union_classifier() -> Pipeline:
    pipe1 = Pipeline(
        [
            ("select", Select(["sepal length (cm)", "sepal width (cm)"])),
            ("scale", DFStandardScaler()),
        ]
    )
    pipe2 = Pipeline(
        [
            ("select", Select(["petal length (cm)", "petal width (cm)"])),
            ("scale", DFStandardScaler()),
        ]
    )
    union = DFFeatureUnion(transformer_list=[("pipe1", pipe1), ("pipe2", pipe2)])
    return Pipeline(
        [("features", union), ("estimator", LogisticRegression(solver="liblinear"))]
    )


@pytest.fixture(scope="session")
def pipeline_forest_classifier() -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", DFStandardScaler()),
            ("estimator", RandomForestClassifier(n_estimators=10)),
        ]
    )
    return pipe


@pytest.fixture()
def estimator_pickle_path_factory(
    train_iris_dataset, tmp_path: pathlib.Path
) -> Callable[[str], pathlib.Path]:
    def tmp_estimator_pickle_path(filename: str) -> pathlib.Path:
        file_path = tmp_path / filename
        model = Model(LogisticRegression(solver="liblinear"))
        model.score_estimator(train_iris_dataset)
        joblib.dump(model.estimator, file_path)
        return pathlib.Path(file_path)

    return tmp_estimator_pickle_path


@pytest.fixture()
def open_estimator_pickle(
    estimator_pickle_path_factory: Callable[[str], pathlib.Path],
    request: FixtureRequest,
):
    def tmp_open_estimator_pickle(path: Optional[pathlib.Path] = None):
        if path is None:
            f = estimator_pickle_path_factory("tmp.pkl").open(mode="rb")
        else:
            f = path.open(mode="rb")
        request.addfinalizer(f.close)
        return f

    return tmp_open_estimator_pickle
