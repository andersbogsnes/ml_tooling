import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.logging import Log
from ml_tooling.metrics import Metrics, Metric
from ml_tooling.result import Result
from ml_tooling.search.gridsearch import _fit_gridpoint
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import MLToolingError


class TestBaseClass:
    def test_is_properties_works(self, classifier, regression, pipeline_linear):
        assert classifier.is_regressor is False
        assert classifier.is_classifier is True
        assert regression.is_regressor is True
        assert regression.is_classifier is False
        assert classifier.is_pipeline is False
        assert regression.is_pipeline is False

        pipeline = Model(pipeline_linear)
        assert pipeline.is_pipeline is True

    def test_can_use_default_metric(self, test_dataset):
        model = Model(LogisticRegression())
        result = model.score_estimator(test_dataset)

        assert result.metrics.metric == "accuracy"

    def test_can_use_specified_metric(self, test_dataset):
        model = Model(LogisticRegression())
        result = model.score_estimator(test_dataset, metrics="roc_auc")

        assert result.metrics.metric == "roc_auc"

    def test_can_use_multiple_metrics(self, test_dataset):
        model = Model(LogisticRegression())
        result = model.score_estimator(test_dataset, metrics=["accuracy", "roc_auc"])

        assert len(result.metrics) == 2
        assert "accuracy" in result.metrics
        assert "roc_auc" in result.metrics

    def test_instantiate_model_with_non_estimator_pipeline_fails(self, base):
        example_pipe = Pipeline([("scale", DFStandardScaler)])
        with pytest.raises(
            MLToolingError,
            match="You passed a Pipeline without an estimator as the last step",
        ):
            base(example_pipe)

    def test_instantiate_model_with_other_object_fails(self, base):
        with pytest.raises(
            MLToolingError,
            match=f"Expected a Pipeline or Estimator - got <class 'dict'>",
        ):
            base({})

    def test_make_prediction_errors_when_model_is_not_fitted(self, base, test_dataset):
        with pytest.raises(MLToolingError, match="You haven't fitted the estimator"):
            model = base(LinearRegression())
            model.make_prediction(test_dataset, 5)

    def test_make_prediction_errors_if_asked_for_proba_without_predict_proba_method(
        self, base, test_dataset
    ):
        with pytest.raises(
            MLToolingError, match="LinearRegression does not have a `predict_proba`"
        ):
            model = base(LinearRegression())
            model.train_estimator(test_dataset)
            model.make_prediction(test_dataset, 5, proba=True)

    @pytest.mark.parametrize("use_index, expected_index", [(False, 0), (True, 5)])
    def test_make_prediction_returns_prediction_if_proba_is_false(
        self, classifier, use_index, expected_index, test_dataset
    ):
        results = classifier.make_prediction(
            test_dataset, 5, proba=False, use_index=use_index
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
        self, classifier, use_index, expected_index, test_dataset
    ):
        results = classifier.make_prediction(
            test_dataset, 5, proba=True, use_index=use_index
        )
        assert isinstance(results, pd.DataFrame)
        assert 2 == results.ndim
        assert np.all((results <= 1) & (results >= 0))
        assert np.all(np.sum(results, axis=1) == 1)
        assert results.index == pd.RangeIndex(
            start=expected_index, stop=expected_index + 1, step=1
        )

    def test_score_estimator_fails_if_no_train_test_data_available(
        self, base, base_dataset
    ):
        model = base(LinearRegression())

        with pytest.raises(MLToolingError, match="Must run create_train_test first!"):
            model.score_estimator(base_dataset())

    def test_default_metric_getter_works_as_expected_classifier(self, base):
        rf = base(RandomForestClassifier(n_estimators=10))
        assert rf.config.CLASSIFIER_METRIC == "accuracy"
        assert rf.config.REGRESSION_METRIC == "r2"
        assert rf.default_metric == "accuracy"
        rf.default_metric = "fowlkes_mallows_score"
        assert rf.config.CLASSIFIER_METRIC == "fowlkes_mallows_score"
        assert rf.config.REGRESSION_METRIC == "r2"
        assert rf.default_metric == "fowlkes_mallows_score"

    def test_default_metric_getter_works_as_expected_regressor(self, base):
        linreg = base(LinearRegression())
        assert linreg.config.CLASSIFIER_METRIC == "accuracy"
        assert linreg.config.REGRESSION_METRIC == "r2"
        assert linreg.default_metric == "r2"
        linreg.default_metric = "neg_mean_squared_error"
        assert linreg.config.CLASSIFIER_METRIC == "accuracy"
        assert linreg.config.REGRESSION_METRIC == "neg_mean_squared_error"
        assert linreg.default_metric == "neg_mean_squared_error"

    def test_default_metric_works_as_expected_without_pipeline(self, base):
        rf = base(RandomForestClassifier(n_estimators=10))
        linreg = base(LinearRegression())
        assert "accuracy" == rf.default_metric
        assert "r2" == linreg.default_metric
        rf.config.CLASSIFIER_METRIC = "fowlkes_mallows_score"
        linreg.config.REGRESSION_METRIC = "neg_mean_squared_error"
        assert "fowlkes_mallows_score" == rf.default_metric
        assert "neg_mean_squared_error" == linreg.default_metric

    def test_default_metric_works_as_expected_with_pipeline(
        self, base, pipeline_logistic, pipeline_linear
    ):
        logreg = base(pipeline_logistic)
        linreg = base(pipeline_linear)
        assert "accuracy" == logreg.default_metric
        assert "r2" == linreg.default_metric
        logreg.config.CLASSIFIER_METRIC = "fowlkes_mallows_score"
        linreg.config.REGRESSION_METRIC = "neg_mean_squared_error"
        assert "fowlkes_mallows_score" == logreg.default_metric
        assert "neg_mean_squared_error" == linreg.default_metric

    def test_train_model_sets_result_to_none(self, regression, test_dataset):
        assert regression.result is not None
        regression.train_estimator(test_dataset)
        assert regression.result is None

    def test_train_model_followed_by_score_model_returns_correctly(
        self, base, pipeline_logistic, test_dataset
    ):
        model = base(pipeline_logistic)
        model.train_estimator(test_dataset)
        model.score_estimator(test_dataset)

        assert isinstance(model.result, Result)

    def test_model_selection_works_as_expected(self, base, test_dataset):
        models = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=10),
        ]
        best_model, results = base.test_estimators(test_dataset, models)
        assert models[1] is best_model.estimator
        assert 2 == len(results)
        assert results[0].metrics[0].score >= results[1].metrics[0].score
        for result in results:
            assert isinstance(result, Result)

    def test_model_selection_with_nonstandard_metric_works_as_expected(
        self, base, test_dataset
    ):
        estimators = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=10),
        ]
        best_estimator, results = base.test_estimators(
            test_dataset, estimators, metrics="roc_auc"
        )
        for result in results:
            assert "roc_auc" in result.metrics

    def test_model_selection_with_pipeline_works_as_expected(
        self, base, pipeline_logistic, pipeline_dummy_classifier, test_dataset
    ):
        estimators = [pipeline_logistic, pipeline_dummy_classifier]
        best_estimator, results = base.test_estimators(test_dataset, estimators)

        for result in results:
            assert (
                result.model.estimator_name
                == result.model.estimator.steps[-1][1].__class__.__name__
            )

        assert best_estimator.estimator == estimators[0]

    def test_regression_model_can_be_saved(
        self, classifier, tmp_path, base, test_dataset
    ):
        expected_path = tmp_path / "test_model.pkl"

        classifier.score_estimator(test_dataset)
        classifier.save_estimator(expected_path)

        assert expected_path.exists()

        loaded_model = base.load_estimator(str(expected_path))
        assert loaded_model.estimator.get_params() == classifier.estimator.get_params()

    def test_save_model_saves_pipeline_correctly(
        self, base, pipeline_logistic, tmp_path, test_dataset
    ):

        save_dir = tmp_path / "test_model_1.pkl"
        model = base(pipeline_logistic)
        model.train_estimator(test_dataset)
        model.save_estimator(save_dir)
        assert save_dir.exists()

    def test_save_model_saves_logging_dir_correctly(
        self, classifier: Model, tmp_path: pathlib.Path, monkeypatch
    ):
        def mockreturn():
            return "1234"

        monkeypatch.setattr("ml_tooling.logging.log_estimator.get_git_hash", mockreturn)
        save_dir = tmp_path / "estimator"
        expected_file = save_dir / "test_model3.pkl"
        with classifier.log(str(save_dir)):
            classifier.save_estimator(expected_file)

        assert expected_file.exists()
        assert (
            "LogisticRegression" in [str(file) for file in save_dir.rglob("*.yaml")][0]
        )

    def test_save_model_errors_if_path_is_dir(self, classifier, tmp_path):
        with pytest.raises(MLToolingError, match=f"Passed directory {tmp_path}"):
            classifier.save_estimator(tmp_path)

    def test_gridsearch_model_returns_as_expected(
        self, base, pipeline_logistic, test_dataset
    ):
        model = base(pipeline_logistic)
        model, results = model.gridsearch(
            test_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

    def test_gridsearch_model_does_not_fail_when_run_twice(
        self, base, pipeline_logistic, test_dataset
    ):
        model = base(pipeline_logistic)
        best_model, results = model.gridsearch(
            test_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

        best_model, results = model.gridsearch(
            test_dataset, param_grid={"clf__penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model.estimator, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, Result)

    def test_fit_gridpoint_returns_new_estimator(self, test_dataset):
        estimator = _fit_gridpoint(
            LogisticRegression(),
            params={"penalty": "l2"},
            train_x=test_dataset.train_x,
            train_y=test_dataset.train_y,
        )

        assert estimator.get_params()["penalty"] == "l2"
        assert estimator.coef_ is not None
        assert isinstance(estimator, LogisticRegression)

    def test_log_context_manager_works_as_expected(self, regression):
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
        self, tmpdir, base, test_dataset
    ):
        model = base(LinearRegression())

        runs = tmpdir.mkdir("runs")
        with model.log(runs):
            result = model.score_estimator(test_dataset)

        for file in runs.visit("LinearRegression_*"):
            with open(file) as f:
                log_result = yaml.safe_load(f)

            assert result.score == log_result["metrics"]["r2"]
            assert result.estimator_name == log_result["estimator_name"]

    def test_log_context_manager_logs_when_gridsearching(
        self, tmpdir, base, test_dataset
    ):
        model = base(LinearRegression())
        runs = tmpdir.mkdir("runs")
        with model.log(runs):
            _, result = model.gridsearch(test_dataset, {"normalize": [True, False]})

        for file in runs.visit("LinearRegression_*"):
            with open(file) as f:
                log_result = yaml.safe_load(f)

            model_results = [round(r.score, 4) for r in result]
            assert round(log_result["metrics"]["r2"], 4) in model_results
            assert result.estimator_name == log_result["estimator_name"]

    def test_test_models_logs_when_given_dir(self, tmpdir, base, test_dataset):
        test_models_log = tmpdir.mkdir("test_estimators")
        base.test_estimators(
            test_dataset,
            [RandomForestClassifier(n_estimators=10), DummyClassifier()],
            log_dir=test_models_log,
        )

        for file in test_models_log.visit("*.yaml"):
            with open(file) as f:
                result = yaml.safe_load(f)
                model_name = result["model_name"]
                assert model_name in {
                    "IrisData_RandomForestClassifier",
                    "IrisData_DummyClassifier",
                }

    def test_train_model_errors_correct_when_not_scored(
        self, base, pipeline_logistic, tmp_path, test_dataset
    ):

        model = base(pipeline_logistic)
        with pytest.raises(MLToolingError, match="You haven't scored the estimator"):
            with model.log(tmp_path):
                model.train_estimator(test_dataset)
                model.save_estimator(tmp_path / "test_model4.pkl")

    def test_dump_serializes_correctly_without_pipeline(self, regression):
        serialized_model = regression.to_dict()
        expected = [
            {
                "module": "sklearn.linear_model.base",
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
                "module": "sklearn.preprocessing.data",
                "classname": "StandardScaler",
                "params": {"copy": True, "with_mean": True, "with_std": True},
            },
            {
                "name": "clf",
                "module": "sklearn.linear_model.base",
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

    def test_can_load_serialized_model_from_pipeline(
        self, pipeline_linear, tmp_path: pathlib.Path
    ):
        model = Model(pipeline_linear)
        log = Log(
            name="test",
            estimator=model.to_dict(),
            metrics=Metrics([Metric("accuracy", score=1.0)]),
        )
        save_file = log.save_log(tmp_path)
        model2 = Model.from_yaml(save_file)

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
        save_file = log.save_log(tmp_path)
        model2 = Model.from_yaml(save_file)
        assert model2.estimator.get_params() == classifier.estimator.get_params()

    def test_gridsearch_uses_default_metric(
        self, classifier: Model, test_dataset: Dataset
    ):
        model, results = classifier.gridsearch(
            test_dataset, param_grid={"penalty": ["l1", "l2"]}
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score
        assert results[0].metrics.metric == "accuracy"

        assert isinstance(model, Model)

    def test_gridsearch_can_take_multiple_metrics(
        self, classifier: Model, test_dataset: Dataset
    ):
        model, results = classifier.gridsearch(
            test_dataset,
            param_grid={"penalty": ["l1", "l2"]},
            metrics=["accuracy", "roc_auc"],
        )

        assert len(results) == 2
        assert results[0].metrics.score >= results[1].metrics.score

        for result in results:
            assert len(result.metrics) == 2
            assert "accuracy" in result.metrics
            assert "roc_auc" in result.metrics
            assert result.metrics.metric == "accuracy"
            assert result.metrics.score == result.metrics[0].score
