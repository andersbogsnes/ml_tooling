import pytest
import joblib
import pathlib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.transformers import DFStandardScaler, DFFeatureUnion, Select


@pytest.fixture()
def regression(base: type, test_dataset: Dataset) -> Model:
    model: Model = base(LinearRegression())
    model.score_estimator(test_dataset)
    return model


@pytest.fixture()
def regression_cv(base: type, test_dataset: Dataset) -> Model:
    model: Model = base(LinearRegression())
    model.score_estimator(test_dataset, cv=2)
    return model


@pytest.fixture()
def classifier(base: type, test_dataset: Dataset) -> Model:
    model: Model = base(LogisticRegression(solver="liblinear"))
    model.score_estimator(test_dataset)
    return model


@pytest.fixture()
def classifier_cv(base: type, test_dataset: Dataset) -> Model:
    model: Model = base(LogisticRegression(solver="liblinear"))
    model.score_estimator(test_dataset, cv=2)
    return model


@pytest.fixture
def pipeline_logistic() -> Pipeline:
    pipe = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear"))]
    )
    return pipe


@pytest.fixture
def pipeline_linear() -> Pipeline:
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LinearRegression())])
    return pipe


@pytest.fixture
def pipeline_dummy_classifier() -> Pipeline:
    pipe = Pipeline([("scale", DFStandardScaler()), ("clf", DummyClassifier())])
    return pipe


@pytest.fixture
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
    union = DFFeatureUnion(transformer_list=[('pipe1', pipe1),
                                             ('pipe2', pipe2)])
    return Pipeline([("features", union), ("clf", LogisticRegression(solver="lbfgs"))])


@pytest.fixture
def pipeline_forest_classifier() -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", DFStandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10)),
        ]
    )
    return pipe


@pytest.fixture
def estimator_pickle_path(test_dataset, tmp_path):
    file_path = tmp_path / "tmp.pkl"
    model = Model(LogisticRegression(solver="liblinear"))
    model.score_estimator(test_dataset)
    joblib.dump(model.estimator, file_path)
    return pathlib.Path(file_path)


@pytest.fixture
def open_estimator_pickle(estimator_pickle_path: pathlib.Path, request):
    f = estimator_pickle_path.open(mode="rb")
    request.addfinalizer(f.close)
    return f
