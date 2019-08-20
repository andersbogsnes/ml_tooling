import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_tooling import ModelData
from ml_tooling.result import CVResult, Result
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import MLToolingError


class TestBaseClass:
    def test_ModelData_can_be_instantiated_without_a_model(self, base):
        no_estimator = base()

        assert no_estimator._estimator is None
        assert no_estimator._plotter is None

    def test_ModelData_init_estimator_works_as_expected(self, base):
        no_estimator = base()

        assert no_estimator._estimator is None

        estimator = RandomForestClassifier()
        no_estimator.init_estimator(estimator)

        assert no_estimator.estimator is estimator

    def test_ModelData_raises_when_accessing_estimator(self, base):
        no_estimator = base()
        with pytest.raises(MLToolingError, match="No estimator selected"):
            no_estimator.estimator

    def test_class_name_property_returns_class_name(self, regression):
        reg = regression
        assert reg.class_name == "IrisModel"

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

    def test_make_prediction_errors_when_model_is_not_fitted(self, base):
        with pytest.raises(MLToolingError, match="You haven't fitted the estimator"):
            model = base(LinearRegression())
            model.make_prediction(5)

    def test_make_prediction_errors_if_asked_for_proba_without_predict_proba_method(
        self, base
    ):
        with pytest.raises(
            MLToolingError, match="LinearRegression does not have a `predict_proba`"
        ):
            model = base(LinearRegression())
            model.train_estimator()
            model.make_prediction(5, proba=True)

    @pytest.mark.parametrize("use_index, expected_index", [(False, 0), (True, 5)])
    def test_make_prediction_returns_prediction_if_proba_is_false(
        self, classifier, use_index, expected_index
    ):
        results = classifier.make_prediction(5, proba=False, use_index=use_index)
        assert isinstance(results, pd.DataFrame)
        assert 2 == results.ndim
        assert np.all((results == 1) | (results == 0))
        assert np.all(np.sum(results, axis=1) == 0)
        assert results.index == pd.RangeIndex(
            start=expected_index, stop=expected_index + 1, step=1
        )

    @pytest.mark.parametrize("use_index, expected_index", [(False, 0), (True, 5)])
    def test_make_prediction_returns_proba_if_proba_is_true(
        self, classifier, use_index, expected_index
    ):
        results = classifier.make_prediction(5, proba=True, use_index=use_index)
        assert isinstance(results, pd.DataFrame)
        assert 2 == results.ndim
        assert np.all((results <= 1) & (results >= 0))
        assert np.all(np.sum(results, axis=1) == 1)
        assert results.index == pd.RangeIndex(
            start=expected_index, stop=expected_index + 1, step=1
        )

    def test_train_model_saves_x_and_y_as_expected(self, regression):
        expected_x, expected_y = regression.get_training_data()
        regression.train_estimator()
        assert np.all(expected_x == regression.data.x)
        assert np.all(expected_y == regression.data.y)

    def test_default_metric_getter_works_as_expected_classifer(self, base):
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

    def test_train_model_sets_result_to_none(self, regression):
        assert regression.result is not None
        regression.train_estimator()
        assert regression.result is None

    def test_train_model_followed_by_score_model_returns_correctly(
        self, base, pipeline_logistic
    ):
        model = base(pipeline_logistic)
        model.train_estimator()
        model.score_estimator()

        assert isinstance(model.result, Result)

    def test_model_selection_works_as_expected(self, base):
        models = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=10),
        ]
        best_model, results = base.test_estimators(models)
        assert models[1] is best_model.estimator
        assert 2 == len(results)
        assert results[0].score >= results[1].score
        for result in results:
            assert isinstance(result, Result)

    def test_model_selection_with_nonstandard_metric_works_as_expected(self, base):
        estimators = [
            LogisticRegression(solver="liblinear"),
            RandomForestClassifier(n_estimators=10),
        ]
        best_estimator, results = base.test_estimators(estimators, metric="roc_auc")
        for result in results:
            assert result.metric == "roc_auc"

    def test_model_selection_with_pipeline_works_as_expected(
        self, base, pipeline_logistic, pipeline_dummy_classifier
    ):
        estimators = [pipeline_logistic, pipeline_dummy_classifier]
        best_estimator, results = base.test_estimators(estimators)

        for result in results:
            assert (
                result.estimator_name
                == result.estimator.steps[-1][1].__class__.__name__
            )

        assert best_estimator.estimator == estimators[0]

    @pytest.mark.parametrize(
        "sub_folder, filename",
        [
            (True, None),
            (True, "saved_model.pkl"),
            (False, None),
            (False, "saved_model.pkl"),
        ],
    )
    def test_regression_model_can_be_saved(
        self, classifier, tmpdir, base, monkeypatch, sub_folder, filename
    ):
        def mockreturn():
            return "1234"

        monkeypatch.setattr("ml_tooling.baseclass.get_git_hash", mockreturn)

        path = tmpdir.join("sub_folder") if sub_folder else tmpdir
        expected_filename = (
            "IrisModel_LogisticRegression_1234.pkl" if not filename else filename
        )

        classifier.score_estimator()
        classifier.save_estimator(path, filename=filename)

        expected_path = path.join(expected_filename)
        assert expected_path.check()

        loaded_model = base.load_estimator(str(expected_path))
        assert loaded_model.estimator.get_params() == classifier.estimator.get_params()

    def test_save_model_saves_pipeline_correctly(
        self, base, pipeline_logistic, monkeypatch, tmpdir
    ):
        def mockreturn():
            return "1234"

        monkeypatch.setattr("ml_tooling.baseclass.get_git_hash", mockreturn)
        save_dir = tmpdir.mkdir("estimator")
        model = base(pipeline_logistic)
        model.train_estimator()
        model.save_estimator(save_dir)
        expected_name = "IrisModel_LogisticRegression_1234.pkl"
        assert save_dir.join(expected_name).check()

    def test_save_model_saves_logging_dir_correctly(
        self, classifier, tmpdir, monkeypatch
    ):
        def mockreturn():
            return "1234"

        monkeypatch.setattr("ml_tooling.baseclass.get_git_hash", mockreturn)
        save_dir = tmpdir.mkdir("estimator")
        with classifier.log(save_dir):
            classifier.save_estimator(save_dir)

        expected_name = "IrisModel_LogisticRegression_1234.pkl"
        assert save_dir.join(expected_name).check()
        assert (
            "LogisticRegression" in [str(file) for file in save_dir.visit("*.yaml")][0]
        )

    def test_setup_model_raises_not_implemented_error(self, base):
        with pytest.raises(NotImplementedError):
            base.setup_estimator()

    def test_setup_model_works_when_implemented(self):
        class DummyModel(ModelData):
            def get_prediction_data(self, idx):
                pass

            def get_training_data(self):
                pass

            @classmethod
            def setup_estimator(cls):
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(solver="lbgfs")),
                    ]
                )
                return cls(pipeline)

        model = DummyModel.setup_estimator()
        assert model.estimator_name == "LogisticRegression"
        assert hasattr(model, "coef_") is False

    def test_gridsearch_model_returns_as_expected(self, base, pipeline_logistic):
        model = base(pipeline_logistic)
        model, results = model.gridsearch(param_grid={"penalty": ["l1", "l2"]})
        assert isinstance(model, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, CVResult)

    def test_gridsearch_model_does_not_fail_when_run_twice(
        self, base, pipeline_logistic
    ):
        model = base(pipeline_logistic)
        best_model, results = model.gridsearch(param_grid={"penalty": ["l1", "l2"]})
        assert isinstance(best_model, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, CVResult)

        best_model, results = model.gridsearch(param_grid={"penalty": ["l1", "l2"]})
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

    def test_log_context_manager_logs_when_scoring_model(self, tmpdir, base):
        model = base(LinearRegression())

        runs = tmpdir.mkdir("runs")
        with model.log(runs):
            result = model.score_estimator()

        for file in runs.visit("LinearRegression_*"):
            with open(file) as f:
                log_result = yaml.safe_load(f)

            assert result.score == log_result["metrics"]["r2"]
            assert result.estimator_name == log_result["estimator_name"]

    def test_log_context_manager_logs_when_gridsearching(self, tmpdir, base):
        model = base(LinearRegression())
        runs = tmpdir.mkdir("runs")
        with model.log(runs):
            _, result = model.gridsearch({"normalize": [True, False]})

        for file in runs.visit("LinearRegression_*"):
            with open(file) as f:
                log_result = yaml.safe_load(f)

            model_results = [round(r.score, 4) for r in result]
            assert round(log_result["metrics"]["r2"], 4) in model_results
            assert result.estimator_name == log_result["estimator_name"]

    def test_test_models_logs_when_given_dir(self, tmpdir, base):
        test_models_log = tmpdir.mkdir("test_estimators")
        base.test_estimators(
            [RandomForestClassifier(n_estimators=10), DummyClassifier()],
            log_dir=test_models_log,
        )

        for file in test_models_log.visit("*.yaml"):
            with open(file) as f:
                result = yaml.safe_load(f)
                model_name = result["estimator_name"]
                assert model_name in {"RandomForestClassifier", "DummyClassifier"}

    def test_train_model_errors_correct_when_not_scored(
        self, base, pipeline_logistic, tmpdir
    ):
        model = base(pipeline_logistic)
        with pytest.raises(MLToolingError, match="You haven't scored the estimator"):
            with model.log(tmpdir):
                model.train_estimator()
                model.save_estimator(tmpdir)

    def test_models_share_data(self):
        class test_class(ModelData):
            def get_training_data(self):
                return (
                    pd.DataFrame({"a": [1, 2, 3, 3, 2, 4]}),
                    pd.Series([0, 1, 1, 0, 1, 0]),
                )

            def get_prediction_data(self, *args):
                pass

        model1 = test_class(LogisticRegression(solver="lbfgs"))
        model2 = test_class(LogisticRegression(solver="lbfgs"))
        pd.testing.assert_frame_equal(model1.data.x, model2.data.x)

        model1.data.x = pd.DataFrame({"b": [9, 9, 8, 9, 2, 4]})
        pd.testing.assert_frame_equal(model1.data.x, model2.data.x)

    def test_classes_do_not_share_data(self):
        pd_df1 = pd.DataFrame({"a": [1, 2, 3, 3, 2, 4]})
        pd_s1 = pd.Series([0, 1, 1, 0, 1, 0])
        pd_df2 = pd.DataFrame({"b": [9, 9, 8, 9, 2, 4]})

        class class1(ModelData):
            def get_training_data(self):
                return pd_df1, pd_s1

            def get_prediction_data(self, *args):
                pass

        cl1 = class1(LogisticRegression(solver="lbfgs"))
        cl1.score_estimator()

        class class2(ModelData):
            def get_training_data(self):
                return pd_df1, pd_s1

            def get_prediction_data(self, *args):
                pass

        cl2 = class2(LogisticRegression(solver="lbfgs"))
        cl2.score_estimator()

        pd.testing.assert_frame_equal(cl1.data.x, cl2.data.x)
        cl2.data.x = pd_df2

        pd.testing.assert_frame_equal(cl1.data.x, pd_df1)
        pd.testing.assert_frame_equal(cl2.data.x, pd_df2)

    def test_scoring_estimator_with_no_estimator_raises(self, base):
        clf = base()
        with pytest.raises(
            MLToolingError,
            match="No estimator selected. " "Use .init_estimator to set an estimator",
        ):
            clf.score_estimator()

    def test_training_estimator_with_no_estimator_raises(self, base):
        clf = base()
        with pytest.raises(
            MLToolingError,
            match="No estimator selected. " "Use .init_estimator to set an estimator",
        ):
            clf.train_estimator()

    def test_gridsearching_estimator_with_no_estimator_raises(self, base):
        clf = base()
        with pytest.raises(
            MLToolingError,
            match="No estimator selected. " "Use .init_estimator to set an estimator",
        ):
            clf.gridsearch({"some_params": [1, 2, 3]})

    def test_saving_estimator_with_no_estimator_raises(self, base):
        clf = base()
        with pytest.raises(
            MLToolingError,
            match="No estimator selected. " "Use .init_estimator to set an estimator",
        ):
            clf.save_estimator()
