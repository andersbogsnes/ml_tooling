import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy.stats as stats
import pytest
import yaml
import datetime

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import loguniform

from ml_tooling.storage import FileStorage
from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.logging import Log
from ml_tooling.metrics import Metrics, Metric
from ml_tooling.result import Result
from ml_tooling.search.gridsearch import prepare_gridsearch_estimators
from ml_tooling.search.randomsearch import prepare_randomsearch_estimators
from ml_tooling.transformers import DFStandardScaler, DFFeatureUnion
from ml_tooling.utils import MLToolingError, DatasetError


class TestBaseClass:
    def test_is_properties_works(
        self, classifier: Model, regression: Model, pipeline_linear: Pipeline
    ):
        assert classifier.is_regressor is False
        assert classifier.is_classifier is True
        assert regression.is_regressor is True
        assert regression.is_classifier is False
        assert classifier.is_pipeline is False
        assert regression.is_pipeline is False

        pipeline = Model(pipeline_linear)
        assert pipeline.is_pipeline is True

    def test_instantiate_model_with_non_estimator_pipeline_fails(self):
        example_pipe = Pipeline([("scale", DFStandardScaler)])
        with pytest.raises(
            MLToolingError,
            match="You passed a Pipeline without an estimator as the last step",
        ):
            Model(example_pipe)

    def test_instantiate_model_with_other_object_fails(self):
        with pytest.raises(
            MLToolingError,
            match="Expected a Pipeline or Estimator - got <class 'dict'>",
        ):
            Model({})

    def test_default_metric_getter_works_as_expected_classifier(self):
        rf = Model(RandomForestClassifier(n_estimators=10))
        assert rf.config.CLASSIFIER_METRIC == "accuracy"
        assert rf.config.REGRESSION_METRIC == "r2"
        assert rf.default_metric == "accuracy"
        rf.default_metric = "fowlkes_mallows_score"
        assert rf.config.CLASSIFIER_METRIC == "fowlkes_mallows_score"
        assert rf.config.REGRESSION_METRIC == "r2"
        assert rf.default_metric == "fowlkes_mallows_score"
        rf.reset_config()

    def test_default_metric_getter_works_as_expected_regressor(self):
        linreg = Model(LinearRegression())
        assert linreg.config.CLASSIFIER_METRIC == "accuracy"
        assert linreg.config.REGRESSION_METRIC == "r2"
        assert linreg.default_metric == "r2"
        linreg.default_metric = "neg_mean_squared_error"
        assert linreg.config.CLASSIFIER_METRIC == "accuracy"
        assert linreg.config.REGRESSION_METRIC == "neg_mean_squared_error"
        assert linreg.default_metric == "neg_mean_squared_error"
        linreg.reset_config()

    def test_default_metric_works_as_expected_without_pipeline(self):
        rf = Model(RandomForestClassifier(n_estimators=10))
        linreg = Model(LinearRegression())
        assert "accuracy" == rf.default_metric
        assert "r2" == linreg.default_metric
        rf.config.CLASSIFIER_METRIC = "fowlkes_mallows_score"
        linreg.config.REGRESSION_METRIC = "neg_mean_squared_error"
        assert "fowlkes_mallows_score" == rf.default_metric
        assert "neg_mean_squared_error" == linreg.default_metric
        rf.reset_config()
        linreg.reset_config()

    def test_default_metric_works_as_expected_with_pipeline(
        self, pipeline_logistic: Pipeline, pipeline_linear: Pipeline
    ):
        logreg = Model(pipeline_logistic)
        linreg = Model(pipeline_linear)
        assert "accuracy" == logreg.default_metric
        assert "r2" == linreg.default_metric
        logreg.config.CLASSIFIER_METRIC = "fowlkes_mallows_score"
        linreg.config.REGRESSION_METRIC = "neg_mean_squared_error"
        assert "fowlkes_mallows_score" == logreg.default_metric
        assert "neg_mean_squared_error" == linreg.default_metric
        logreg.reset_config()
        linreg.reset_config()

    def test_regression_model_can_be_saved(
        self, classifier: Model, tmp_path: pathlib.Path, train_iris_dataset
    ):
        classifier.score_estimator(train_iris_dataset)
        load_storage = FileStorage(tmp_path)

        storage = FileStorage(tmp_path)
        saved_model_path = classifier.save_estimator(storage)
        assert saved_model_path.exists()
        loaded_model = classifier.load_estimator(load_storage, saved_model_path)
        assert loaded_model.estimator.get_params() == classifier.estimator.get_params()

    def test_regression_model_filename_is_generated_correctly(
        self, classifier: Model, tmp_path: pathlib.Path, train_iris_dataset
    ):
        storage = FileStorage(tmp_path)
        saved_model_path = classifier.save_estimator(storage)
        assert saved_model_path.exists()
        assert datetime.datetime.strptime(
            saved_model_path.stem, f"{classifier.estimator_name}_%Y-%m-%d_%H:%M:%S.%f"
        )

    def test_save_model_saves_pipeline_correctly(
        self, pipeline_logistic: Pipeline, tmp_path: pathlib.Path, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        model.train_estimator(train_iris_dataset)
        saved_model_path = model.save_estimator(FileStorage(tmp_path))
        assert saved_model_path.exists()

    @patch("ml_tooling.logging.log_estimator.get_git_hash")
    def test_save_estimator_saves_logging_dir_correctly(
        self, mock_hash: MagicMock, classifier: Model, tmp_path: pathlib.Path
    ):
        mock_hash.return_value = "1234"

        with classifier.log(tmp_path):
            expected_file = classifier.save_estimator(FileStorage(tmp_path))

        assert expected_file.exists()
        assert (
            "LogisticRegression" in [str(file) for file in tmp_path.rglob("*.yaml")][0]
        )
        mock_hash.assert_called_once()

    def test_save_estimator_with_prod_flag_saves_correctly(self, classifier: Model):
        mock_storage = MagicMock()
        classifier.save_estimator(mock_storage, prod=True)

        mock_storage.save.assert_called_once_with(
            classifier.estimator, "production_model.pkl", prod=True
        )

    def test_save_estimator_uses_default_storage_if_no_storage_is_passed(
        self, tmp_path: pathlib.Path, classifier: Model
    ):
        classifier.config.ESTIMATOR_DIR = tmp_path
        classifier.save_estimator()

        models = classifier.config.default_storage.get_list()
        assert len(models) == 1
        new_classifier = Model.load_estimator(
            classifier.config.default_storage, models[0]
        )
        assert (
            classifier.estimator.get_params() == new_classifier.estimator.get_params()
        )

    @patch("ml_tooling.baseclass.import_path")
    def test_can_load_production_estimator(
        self, mock_path: MagicMock, open_estimator_pickle
    ):
        mock_path.return_value.__enter__.return_value = open_estimator_pickle()
        model = Model.load_production_estimator("test")
        assert isinstance(model, Model)
        assert isinstance(model.estimator, BaseEstimator)

    def test_log_context_manager_works_as_expected(self, regression: Model):
        assert regression.config.LOG is False
        assert "runs" == regression.config.RUN_DIR.name
        with regression.log("test"):
            assert regression.config.LOG is True
            assert "test" == regression.config.RUN_DIR.name
            assert "runs" == regression.config.RUN_DIR.parent.name

        assert regression.config.LOG is False
        assert "runs" == regression.config.RUN_DIR.name
        assert "test" not in regression.config.RUN_DIR.parts

    def test_log_context_manager_logs_when_scoring_model(
        self, tmp_path: pathlib.Path, train_iris_dataset
    ):
        model = Model(LinearRegression())

        runs = tmp_path / "runs"
        with model.log(str(runs)):
            result = model.score_estimator(train_iris_dataset)

        for file in runs.rglob("LinearRegression_*"):
            with file.open() as f:
                log_result = yaml.safe_load(f)

            assert result.metrics.score == log_result["metrics"]["r2"]
            assert result.model.estimator_name == log_result["estimator_name"]

    def test_test_models_logs_when_given_dir(
        self, tmp_path: pathlib.Path, train_iris_dataset
    ):
        test_models_log = tmp_path / "test_estimators"
        Model.test_estimators(
            train_iris_dataset,
            [
                RandomForestClassifier(n_estimators=10),
                DummyClassifier(strategy="prior"),
            ],
            log_dir=str(test_models_log),
            metrics="accuracy",
        )

        for file in test_models_log.rglob("*.yaml"):
            with file.open() as f:
                result = yaml.safe_load(f)
                model_name = result["model_name"]
                assert model_name in {
                    "IrisData_RandomForestClassifier",
                    "IrisData_DummyClassifier",
                }

    def test_dump_serializes_correctly_without_pipeline(self, regression: Model):
        serialized_model = regression.to_dict()
        expected = [
            {
                "module": "sklearn.linear_model._base",
                "classname": "LinearRegression",
                "params": {
                    "copy_X": True,
                    "fit_intercept": True,
                    "n_jobs": None,
                    "normalize": False,
                },
            }
        ]

        assert serialized_model == expected

    def test_dump_serializes_correctly_with_pipeline(self, pipeline_linear: Pipeline):
        serialized_model = Model(pipeline_linear).to_dict()
        expected = [
            {
                "name": "scale",
                "module": "sklearn.preprocessing._data",
                "classname": "StandardScaler",
                "params": {"copy": True, "with_mean": True, "with_std": True},
            },
            {
                "name": "clf",
                "module": "sklearn.linear_model._base",
                "classname": "LinearRegression",
                "params": {
                    "copy_X": True,
                    "fit_intercept": True,
                    "n_jobs": None,
                    "normalize": False,
                },
            },
        ]

        assert serialized_model == expected

    def test_to_dict_serializes_correctly_with_feature_union(
        self, feature_union_classifier: DFFeatureUnion
    ):
        model = Model(feature_union_classifier)
        result = model.to_dict()
        assert len(result) == 2
        union = result[0]
        assert union["name"] == "features"
        assert len(union["params"]) == 2
        pipe1 = union["params"][0]
        pipe2 = union["params"][1]

        assert pipe1["name"] == "pipe1"
        select1 = pipe1["params"][0]
        scale1 = pipe1["params"][1]

        assert select1["name"] == "select"
        assert select1["classname"] == "Select"
        assert select1["params"] == {
            "columns": ["sepal length (cm)", "sepal width (cm)"]
        }

        assert scale1["name"] == "scale"
        assert scale1["classname"] == "DFStandardScaler"
        assert scale1["params"] == {"copy": True, "with_mean": True, "with_std": True}

        assert pipe2["name"] == "pipe2"
        select2 = pipe2["params"][0]
        scale2 = pipe2["params"][1]

        assert select2["name"] == "select"
        assert select2["classname"] == "Select"
        assert select2["params"] == {
            "columns": ["petal length (cm)", "petal width (cm)"]
        }

        assert scale2["name"] == "scale"
        assert scale2["classname"] == "DFStandardScaler"
        assert scale2["params"] == {"copy": True, "with_mean": True, "with_std": True}

    def test_from_yaml_serializes_correctly_with_feature_union(
        self, feature_union_classifier: DFFeatureUnion, tmp_path: pathlib.Path
    ):

        model = Model(feature_union_classifier)
        result = model.to_dict()

        log = Log(
            name="test", metrics=Metrics.from_list(["accuracy"]), estimator=result
        )
        log.save_log(tmp_path)

        new_model = Model.from_yaml(log.output_path)

        assert len(new_model.estimator.steps[0][1].transformer_list) == 2
        new_steps = new_model.estimator.steps
        old_steps = model.estimator.steps

        assert new_steps[0][0] == old_steps[0][0]
        assert isinstance(new_steps[0][1], type(old_steps[0][1]))

        new_union = new_steps[0][1].transformer_list
        old_union = old_steps[0][1].transformer_list

        assert len(new_union) == len(old_union)

        for new_transform, old_transform in zip(new_union, old_union):
            assert new_transform[1].steps[0][0] == old_transform[1].steps[0][0]
            assert (
                new_transform[1].steps[0][1].get_params()
                == old_transform[1].steps[0][1].get_params()
            )

    def test_can_load_serialized_model_from_pipeline(
        self, pipeline_linear: Pipeline, tmp_path: pathlib.Path
    ):
        model = Model(pipeline_linear)
        log = Log(
            name="test",
            estimator=model.to_dict(),
            metrics=Metrics([Metric("accuracy", score=1.0)]),
        )
        log.save_log(tmp_path)
        model2 = Model.from_yaml(log.output_path)

        for model1, model2 in zip(model.estimator.steps, model2.estimator.steps):
            assert model1[0] == model2[0]
            assert model1[1].get_params() == model2[1].get_params()

    def test_can_load_serialized_model_from_estimator(
        self, classifier: Model, tmp_path: pathlib.Path
    ):
        log = Log(
            name="test",
            estimator=classifier.to_dict(),
            metrics=Metrics([Metric("accuracy", score=1.0)]),
        )
        log.save_log(tmp_path)
        model2 = Model.from_yaml(log.output_path)
        assert model2.estimator.get_params() == classifier.estimator.get_params()


class TestTrainEstimator:
    def test_train_model_sets_result_to_none(
        self, regression: Model, train_iris_dataset
    ):
        assert regression.result is not None
        regression.train_estimator(train_iris_dataset)
        assert regression.result is None

    def test_train_model_followed_by_score_model_returns_correctly(
        self, pipeline_logistic: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        model.train_estimator(train_iris_dataset)
        model.score_estimator(train_iris_dataset)

        assert isinstance(model.result, Result)

    def test_train_model_errors_correctly_when_not_scored(
        self, pipeline_logistic: Pipeline, tmp_path: pathlib.Path, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        with pytest.raises(MLToolingError, match="You haven't scored the estimator"):
            with model.log(str(tmp_path)):
                model.train_estimator(train_iris_dataset)
                model.save_estimator(FileStorage(tmp_path))

    def test_can_score_estimator_with_no_y_value(self):
        class DummyEstimator(BaseEstimator, RegressorMixin):
            def __init__(self):
                self.average = None

            def fit(self, x, y=None):
                self.average = np.mean(x, axis=0)
                return self

            def predict(self, x):
                return self.average

        class DummyData(Dataset):
            def load_training_data(self):
                return pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [4, 5, 6, 7]}), None

            def load_prediction_data(self, *args, **kwargs):
                return pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [4, 5, 6, 7]})

        model = Model(DummyEstimator())
        data = DummyData()
        model.train_estimator(data)

        assert np.all(np.isclose(model.estimator.average, np.array([2.5, 5.5])))

        with pytest.raises(DatasetError, match="The dataset does not define a y value"):
            data.create_train_test()


class TestScoreEstimator:
    def test_score_estimator_if_no_train_test_data_available(
        self, boston_dataset, train_boston_dataset
    ):
        model = Model(LinearRegression())
        data = boston_dataset()

        test = train_boston_dataset
        result = model.score_estimator(test)

        assert model.score_estimator(data) == result

    def test_score_estimator_if_no_train_test_data_available_classification(
        self, iris_dataset, train_iris_dataset
    ):
        model = Model(LogisticRegression())
        data = iris_dataset()

        test = train_iris_dataset
        result = model.score_estimator(test)

        assert model.score_estimator(data) == result

    def test_score_estimator_if_no_train_test_data_available_changed_config(
        self, iris_dataset, train_iris_dataset
    ):
        model = Model(LogisticRegression())
        model.config.RANDOM_STATE = 1
        data = iris_dataset()

        test = iris_dataset()
        test.create_train_test(seed=1)
        result = model.score_estimator(test)

        assert model.score_estimator(data) == result

    def test_can_score_estimator_with_specified_metric(self, train_iris_dataset):
        model = Model(LogisticRegression(solver="liblinear"))
        result = model.score_estimator(train_iris_dataset, metrics="roc_auc")

        assert result.metrics.name == "roc_auc"

    def test_can_score_estimator_with_default_metric(self, train_iris_dataset):
        model = Model(LogisticRegression(solver="liblinear"))
        result = model.score_estimator(train_iris_dataset)

        assert result.metrics.name == "accuracy"

    def test_can_score_estimator_with_multiple_metrics(self, train_iris_dataset):
        model = Model(LogisticRegression(solver="liblinear"))
        result = model.score_estimator(
            train_iris_dataset, metrics=["accuracy", "roc_auc"]
        )

        assert len(result.metrics) == 2
        assert "accuracy" in result.metrics
        assert "roc_auc" in result.metrics


class TestModelSelection:
    def test_model_selection_works_as_expected(self, train_iris_dataset):
        models = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=10),
        ]
        best_model, results = Model.test_estimators(
            train_iris_dataset, models, metrics="accuracy"
        )
        assert models[1] is best_model.estimator
        assert 2 == len(results)
        assert results[0].metrics[0].score >= results[1].metrics[0].score
        for result in results:
            assert isinstance(result, Result)

    def test_model_selection_works_with_default_metric(self, train_iris_dataset):
        models = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=2),
        ]
        best_model, results = Model.test_estimators(train_iris_dataset, models)

        assert models[1] is best_model.estimator
        assert 2 == len(results)
        assert results[0].metrics[0].name == "accuracy"
        assert results[1].metrics[0].name == "accuracy"

    def test_model_selection_works_with_multiple_metrics(self, train_iris_dataset):
        models = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=2),
        ]
        best_model, results = Model.test_estimators(
            train_iris_dataset, models, metrics=["accuracy", "roc_auc"]
        )

        assert models[1] is best_model.estimator
        assert 2 == len(results)
        assert 2 == len(results[0].metrics)
        assert 2 == len(results[1].metrics)

    def test_model_selection_with_nonstandard_metric_works_as_expected(
        self, train_iris_dataset
    ):
        estimators = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=10),
        ]
        best_estimator, results = Model.test_estimators(
            train_iris_dataset, estimators, metrics="roc_auc"
        )
        for result in results:
            assert "roc_auc" in result.metrics

    def test_model_selection_with_pipeline_works_as_expected(
        self,
        pipeline_logistic: Pipeline,
        pipeline_dummy_classifier: Pipeline,
        train_iris_dataset,
    ):
        estimators = [pipeline_logistic, pipeline_dummy_classifier]
        best_estimator, results = Model.test_estimators(
            train_iris_dataset, estimators, "accuracy"
        )

        for result in results:
            assert (
                result.model.estimator_name
                == result.model.estimator.steps[-1][1].__class__.__name__
            )

        assert best_estimator.estimator == estimators[0]

    def test_model_selection_refits_final_model(self, train_iris_dataset):
        estimators = [LogisticRegression(solver="liblinear")]

        model = LogisticRegression(solver="liblinear").fit(
            train_iris_dataset.train_x, train_iris_dataset.train_y
        )
        model2, results2 = Model.test_estimators(
            train_iris_dataset, estimators, cv=2, refit=True, metrics="accuracy"
        )

        assert (model.coef_ == model2.estimator.coef_).all()


class TestGridSearch:
    def test_gridsearch_model_returns_as_expected(
        self, pipeline_logistic: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        model, results = model.gridsearch(
            train_iris_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

    def test_gridsearch_model_does_not_fail_when_run_twice(
        self, pipeline_logistic: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        best_model, results = model.gridsearch(
            train_iris_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

        best_model, results = model.gridsearch(
            train_iris_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

    def test_prepare_gridsearch_estimators_has_different_parameters(self):
        estimators = prepare_gridsearch_estimators(
            LogisticRegression(), params={"penalty": ["l2", "l1"]}
        )

        for estimator, penalty in zip(estimators, ["l2", "l1"]):
            assert estimator.get_params()["penalty"] == penalty
            assert hasattr(estimator, "coef_") is False
            assert isinstance(estimator, LogisticRegression)

    def test_prepare_gridsearch_estimators_in_pipeline_has_different_parameters(self):
        pipe = Pipeline([("scale", DFStandardScaler()), ("clf", LogisticRegression())])

        estimators = prepare_gridsearch_estimators(
            pipe, params={"clf__penalty": ["l2", "l1"]}
        )

        for estimator, penalty in zip(estimators, ["l2", "l1"]):
            clf = estimator["clf"]
            assert clf.get_params()["penalty"] == penalty
            assert hasattr(clf, "coef_") is False
            assert isinstance(clf, LogisticRegression)

    def test_gridsearch_uses_default_metric(
        self, classifier: Model, train_iris_dataset
    ):
        model, results = classifier.gridsearch(
            train_iris_dataset, param_grid={"penalty": ["l1", "l2"]}
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score
        assert results[0].metrics.name == "accuracy"

        assert isinstance(model, Model)

    def test_gridsearch_can_take_multiple_metrics(
        self, classifier: Model, train_iris_dataset
    ):
        model, results = classifier.gridsearch(
            train_iris_dataset,
            param_grid={"penalty": ["l1", "l2"]},
            metrics=["accuracy", "roc_auc"],
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score

        for result in results:
            assert len(result.metrics) == 2
            assert "accuracy" in result.metrics
            assert "roc_auc" in result.metrics
            assert result.metrics.name == "accuracy"
            assert result.metrics.score == result.metrics[0].score

    def test_gridsearch_can_log_with_context_manager(
        self, feature_union_classifier, train_iris_dataset, tmp_path
    ):
        classifier = Model(feature_union_classifier)
        classifier.config.RUN_DIR = tmp_path
        with classifier.log("gridsearch_union_test"):
            _, _ = classifier.gridsearch(
                train_iris_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
            )


class TestRandomSearch:
    def test_randomsearch_model_returns_as_expected(
        self, pipeline_logistic: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        model, results = model.randomsearch(
            train_iris_dataset, param_distributions={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

    def test_randomsearch_model_does_not_fail_when_run_twice(
        self, pipeline_logistic: Pipeline, train_iris_dataset
    ):
        model = Model(pipeline_logistic)
        best_model, results = model.randomsearch(
            train_iris_dataset, param_distributions={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

        best_model, results = model.randomsearch(
            train_iris_dataset, param_distributions={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

    def test_prepare_randomsearch_estimators_has_different_parameters(self):
        estimators = prepare_randomsearch_estimators(
            LogisticRegression(), params={"penalty": ["l2", "l1"]}
        )

        for estimator, penalty in zip(estimators, ["l2", "l1"]):
            assert estimator.get_params()["penalty"] == penalty
            assert hasattr(estimator, "coef_") is False
            assert isinstance(estimator, LogisticRegression)

    def test_prepare_randomsearch_estimators_in_pipeline_has_different_parameters(self):
        pipe = Pipeline([("scale", DFStandardScaler()), ("clf", LogisticRegression())])

        estimators = prepare_randomsearch_estimators(
            pipe, params={"clf__penalty": ["l2", "l1"]}
        )

        for estimator, penalty in zip(estimators, ["l2", "l1"]):
            clf = estimator["clf"]
            assert clf.get_params()["penalty"] == penalty
            assert hasattr(clf, "coef_") is False
            assert isinstance(clf, LogisticRegression)

    def test_randomsearch_uses_default_metric(
        self, classifier: Model, train_iris_dataset
    ):
        model, results = classifier.randomsearch(
            train_iris_dataset, param_distributions={"penalty": ["l1", "l2"]}
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score
        assert results[0].metrics.name == "accuracy"

        assert isinstance(model, Model)

    def test_randomsearch_can_take_dist_objs(
        self, classifier: Model, train_iris_dataset
    ):
        param_dist = {
            "penalty": ["l1", "l2"],
            "l1_ratio": stats.uniform(0, 1),
            "C": loguniform(1e-4, 1e0),
        }
        model, results = classifier.randomsearch(
            train_iris_dataset, param_distributions=param_dist, n_iter=2
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score
        assert results[0].metrics.name == "accuracy"

        assert isinstance(model, Model)

    def test_randomsearch_can_take_multiple_metrics(
        self, classifier: Model, train_iris_dataset
    ):
        model, results = classifier.randomsearch(
            train_iris_dataset,
            param_distributions={"penalty": ["l1", "l2"]},
            metrics=["accuracy", "roc_auc"],
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score

        for result in results:
            assert len(result.metrics) == 2
            assert "accuracy" in result.metrics
            assert "roc_auc" in result.metrics
            assert result.metrics.name == "accuracy"
            assert result.metrics.score == result.metrics[0].score

    def test_randomsearch_can_log_with_context_manager(
        self, feature_union_classifier, train_iris_dataset, tmp_path
    ):
        classifier = Model(feature_union_classifier)
        classifier.config.RUN_DIR = tmp_path
        with classifier.log("randomsearch_union_test"):
            _, _ = classifier.randomsearch(
                train_iris_dataset, param_distributions={"clf__penalty": ["l1", "l2"]}
            )


class TestMakePrediction:
    def test_make_prediction_errors_when_model_is_not_fitted(self, train_iris_dataset):
        model = Model(LinearRegression())
        with pytest.raises(MLToolingError, match="You haven't fitted the estimator"):
            model.make_prediction(train_iris_dataset, 5)

    def test_make_prediction_with_regression_sqldataset_works_as_expected(
        self, boston_sqldataset, loaded_boston_db
    ):
        dataset = boston_sqldataset(loaded_boston_db, schema=None)
        dataset.create_train_test(stratify=False)
        model = Model(LinearRegression())
        model.train_estimator(dataset)

        result = model.make_prediction(dataset, 0)

        assert result.shape == (1, 1)
        assert result.columns.tolist() == ["Prediction"]

    def test_make_prediction_with_classification_sqldataset_works_as_expected(
        self, iris_sqldataset, loaded_iris_db
    ):
        dataset = iris_sqldataset(loaded_iris_db, schema=None)
        dataset.create_train_test()
        model = Model(LogisticRegression(solver="lbfgs"))
        model.train_estimator(dataset)

        result = model.make_prediction(dataset, 0, proba=True)

        assert result.shape == (1, 2)
        assert result.columns.tolist() == ["Probability Class 0", "Probability Class 1"]

    def test_make_prediction_errors_if_asked_for_proba_without_predict_proba_method(
        self, train_iris_dataset
    ):
        with pytest.raises(
            MLToolingError, match="LinearRegression does not have a `predict_proba`"
        ):
            model = Model(LinearRegression())
            model.train_estimator(train_iris_dataset)
            model.make_prediction(train_iris_dataset, 5, proba=True)

    @pytest.mark.parametrize("use_index, expected_index", [(False, 0), (True, 5)])
    def test_make_prediction_returns_prediction_if_proba_is_false(
        self,
        classifier: Model,
        use_index: bool,
        expected_index: int,
        train_iris_dataset,
    ):
        results = classifier.make_prediction(
            train_iris_dataset, 5, proba=False, use_index=use_index
        )
        assert isinstance(results, pd.DataFrame)
        assert 2 == results.ndim
        assert np.all((results == 1) | (results == 0))
        assert np.all(np.sum(results, axis=1) == 0)
        assert results.index == pd.RangeIndex(
            start=expected_index, stop=expected_index + 1, step=1
        )

    @pytest.mark.parametrize("use_index, expected_index", [(False, 0), (True, 5)])
    def test_make_prediction_returns_proba_if_proba_is_true(
        self,
        classifier: Model,
        use_index: bool,
        expected_index: int,
        train_iris_dataset,
    ):
        results = classifier.make_prediction(
            train_iris_dataset, 5, proba=True, use_index=use_index
        )
        assert isinstance(results, pd.DataFrame)
        assert 2 == results.ndim
        assert np.all((results <= 1) & (results >= 0))
        assert np.all(np.sum(results, axis=1) == 1)
        assert results.index == pd.RangeIndex(
            start=expected_index, stop=expected_index + 1, step=1
        )

    def test_make_prediction_uses_cache_if_set(self, classifier, iris_df):
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.x = iris_df.head(5).drop(columns="target")

        result = classifier.make_prediction(mock_dataset, use_cache=True)
        assert len(result) == 5
        mock_dataset._load_prediction_data.assert_not_called()
