import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_tooling.result import CVResult, Result
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import MLToolingError
from ml_tooling import BaseClassModel


class TestBaseClass:
    def test_make_prediction_errors_when_model_is_not_fitted(self, base):
        with pytest.raises(MLToolingError, match="You haven't fitted the model"):
            model = base(LinearRegression())
            model.make_prediction(5)

    def test_make_prediction_errors_if_asked_for_proba_without_predict_proba_method(self, base):
        with pytest.raises(MLToolingError, match="LinearRegression doesn't have a `predict_proba`"):
            model = base(LinearRegression())
            model.train_model()
            model.make_prediction(5, proba=True)

    def test_make_prediction_returns_proba_if_proba_is_true(self, classifier):
        results = classifier.make_prediction(5, proba=True)
        assert isinstance(results, np.ndarray)
        assert 2 == results.ndim
        assert np.all((results <= 1) & (results >= 0))
        assert np.all(np.sum(results, axis=1) == 1)

    def test_train_model_saves_x_and_y_as_expected(self, regression):
        expected_x, expected_y = regression.get_training_data()
        regression.train_model()
        assert np.all(expected_x == regression.data.x)
        assert np.all(expected_y == regression.data.y)

    def test_model_selection_works_as_expected(self, base):
        models = [LogisticRegression(solver='liblinear'), RandomForestClassifier(n_estimators=10)]
        best_model, results = base.test_models(models)
        assert models[1] is best_model.model
        assert 2 == len(results)
        assert results[0].score >= results[1].score
        for result in results:
            assert isinstance(result, Result)

    def test_model_selection_with_nonstandard_metric_works_as_expected(self, base):
        models = [LogisticRegression(solver='liblinear'), RandomForestClassifier(n_estimators=10)]
        best_model, results = base.test_models(models, metric='roc_auc')
        for result in results:
            assert result.metric == 'roc_auc'

    def test_model_selection_with_pipeline_works_as_expected(self,
                                                             base,
                                                             pipeline_logistic,
                                                             pipeline_dummy_classifier):
        models = [pipeline_logistic, pipeline_dummy_classifier]
        best_model, results = base.test_models(models)

        for result in results:
            assert result.model_name == result.model.steps[-1][1].__class__.__name__

        assert best_model.model == models[0]

    def test_regression_model_can_be_saved(self, classifier, tmpdir, base, monkeypatch):
        def mockreturn():
            return '1234'

        monkeypatch.setattr('ml_tooling.baseclass.get_git_hash', mockreturn)
        path = tmpdir.mkdir('model')
        classifier.score_model()
        classifier.save_model(path)
        expected_path = path.join('IrisModel_LogisticRegression_1234.pkl')
        assert expected_path.check()

        loaded_model = base.load_model(str(expected_path))
        assert loaded_model.model.get_params() == classifier.model.get_params()

    def test_save_model_saves_correctly(self, classifier, tmpdir, monkeypatch):
        def mockreturn():
            return '1234'

        monkeypatch.setattr('ml_tooling.baseclass.get_git_hash', mockreturn)
        save_dir = tmpdir.mkdir('model')
        classifier.save_model(save_dir)
        expected_name = 'IrisModel_LogisticRegression_1234.pkl'
        assert save_dir.join(expected_name).check()

    def test_setup_model_raises_not_implemented_error(self, base):
        with pytest.raises(NotImplementedError):
            base.setup_model()

    def test_setup_model_works_when_implemented(self):
        class DummyModel(BaseClassModel):
            def get_prediction_data(self, idx):
                pass

            def get_training_data(self):
                pass

            @classmethod
            def setup_model(cls):
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(solver='lbgfs'))
                ])
                return cls(pipeline)

        model = DummyModel.setup_model()
        assert model.model_name == 'LogisticRegression'
        assert hasattr(model, 'coef_') is False

    def test_gridsearch_model_returns_as_expected(self, base, pipeline_logistic):
        model = base(pipeline_logistic)
        model, results = model.gridsearch(param_grid={'penalty': ['l1', 'l2']})
        assert isinstance(model, Pipeline)
        assert 2 == len(results)

        for result in results:
            assert isinstance(result, CVResult)
