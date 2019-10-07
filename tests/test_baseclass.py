import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from ml_tooling.storage import FileStorage
from ml_tooling.result import CVResult, Result
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import MLToolingError


class TestBaseClass:
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
        assert results[0].score >= results[1].score
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
            test_dataset, estimators, metric="roc_auc"
        )
        for result in results:
            assert result.metric == "roc_auc"

    def test_model_selection_with_pipeline_works_as_expected(
        self, base, pipeline_logistic, pipeline_dummy_classifier, test_dataset
    ):
        estimators = [pipeline_logistic, pipeline_dummy_classifier]
        best_estimator, results = base.test_estimators(test_dataset, estimators)

        for result in results:
            assert (
                result.estimator_name
                == result.estimator.steps[-1][1].__class__.__name__
            )

        assert best_estimator.estimator == estimators[0]

    def test_regression_model_can_be_saved(
        self, classifier, tmp_path, base, test_dataset
    ):
        expected_path = tmp_path / "test_model.pkl"

        classifier.score_estimator(test_dataset)
        load_storage = FileStorage(expected_path)

        with FileStorage(expected_path) as storage:
            classifier.save_estimator(storage)

            assert expected_path.exists()

            loaded_model = base.load_estimator(load_storage)
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
        self, classifier, tmp_path, monkeypatch
    ):
        def mockreturn():
            return "1234"

        monkeypatch.setattr("ml_tooling.logging.log_estimator.get_git_hash", mockreturn)
        save_dir = tmp_path / "estimator"
        expected_file = save_dir / "test_model3.pkl"
        with classifier.log(save_dir):
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
            test_dataset, param_grid={"penalty": ["l1", "l2"]}
        )
        assert isinstance(model, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, CVResult)

    def test_gridsearch_model_does_not_fail_when_run_twice(
        self, base, pipeline_logistic, test_dataset
    ):
        model = base(pipeline_logistic)
        best_model, results = model.gridsearch(
            test_dataset, param_grid={"penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, CVResult)

        best_model, results = model.gridsearch(
            test_dataset, param_grid={"penalty": ["l1", "l2"]}
        )
        assert isinstance(best_model, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, CVResult)

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
                model_name = result["estimator_name"]
                assert model_name in {"RandomForestClassifier", "DummyClassifier"}

    def test_train_model_errors_correct_when_not_scored(
        self, base, pipeline_logistic, tmp_path, test_dataset
    ):

        model = base(pipeline_logistic)
        with pytest.raises(MLToolingError, match="You haven't scored the estimator"):
            with model.log(tmp_path):
                model.train_estimator(test_dataset)
                model.save_estimator(tmp_path / "test_model4.pkl")
