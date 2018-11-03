import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_tooling.result import CVResult, Result
from ml_tooling.utils import MLToolingError
from ml_tooling import BaseClassModel


class TestResult:
    @pytest.mark.parametrize('cv', ['with_cv', 'without_cv'])
    def test_linear_model_returns_a_result(self, regression, regression_cv, cv):
        if cv == 'with_cv':
            result = regression_cv.result
            assert isinstance(result, CVResult)
            assert 2 == len(result.cross_val_scores)
            assert 2 == result.cv
            assert '2-fold Cross-validated' in result.__repr__()
            assert result.model == regression_cv.model
        else:
            result = regression.result
            assert isinstance(result, Result)
            assert hasattr(result, 'cross_val_std') is False
            assert result.model == regression.model

        assert result.score > 0
        assert 'r2' == result.metric
        assert 'LinearRegression' == result.model_name

    @pytest.mark.parametrize('cv', ['with_cv', 'without_cv'])
    def test_regression_model_returns_a_result(self, classifier, classifier_cv, cv):
        if cv == 'with_cv':
            result = classifier_cv.result
            assert isinstance(result, CVResult)
            assert 2 == len(result.cross_val_scores)
            assert 2 == result.cv
            assert '2-fold Cross-validated' in result.__repr__()
            assert result.model == classifier_cv.model

        else:
            result = classifier.result
            assert isinstance(result, Result)
            assert hasattr(result, 'cross_val_std') is False
            assert result.model == classifier.model

        assert result.score > 0
        assert 'accuracy' == result.metric
        assert 'LogisticRegression' == result.model_name

    def test_pipeline_regression_returns_correct_result(self, base, pipeline_linear):
        model = base(pipeline_linear)
        result = model.score_model()
        assert isinstance(result, Result)
        assert 'LinearRegression' == result.model_name
        assert isinstance(result.model, Pipeline)

    def test_pipeline_logistic_returns_correct_result(self, base, pipeline_logistic):
        model = base(pipeline_logistic)
        result = model.score_model()
        assert isinstance(result, Result)
        assert 'LogisticRegression' == result.model_name
        assert isinstance(result.model, Pipeline)

    def test_cvresult_equality_operators(self):
        first_result = CVResult(model=None, model_name='test', cross_val_mean=.7, cross_val_std=.2)
        second_result = CVResult(model=None, model_name='test2', cross_val_mean=.5,
                                 cross_val_std=.2)

        assert first_result > second_result

    def test_result_equality_operators(self):
        first_result = Result(model=None, model_name='test', score=.7)
        second_result = Result(model=None, model_name='test2', score=.5)

        assert first_result > second_result

    def test_max_works_with_cv_result(self):
        first_result = CVResult(model=None, model_name='test', cross_val_mean=.7, cross_val_std=.2)
        second_result = CVResult(model_name='test', model=None, cross_val_mean=.5, cross_val_std=.2)

        max_result = max([first_result, second_result])

        assert first_result is max_result

    def test_max_works_with_result(self):
        first_result = Result(model=None, model_name='test', score=.7)
        second_result = Result(model_name='test', model=None, score=.5)

        max_result = max([first_result, second_result])

        assert first_result is max_result


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
