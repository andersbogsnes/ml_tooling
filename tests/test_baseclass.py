import pytest

from ml_utils.baseclass._baseclass import Result, MLUtilsError
from ml_utils import BaseClassModel
from sklearn.linear_model import LinearRegression
import numpy as np
import os


def test_can_change_config():
    class SomeModel(BaseClassModel):
        def get_training_data(self):
            pass

        def get_prediction_data(self, *args):
            pass

    test_model = SomeModel(LinearRegression())
    assert 10 == test_model.config["CROSS_VALIDATION"]
    test_model.set_config({"CROSS_VALIDATION": 2})
    assert test_model.config["CROSS_VALIDATION"] == 2


def test_linear_model_returns_a_result(regression):
    result = regression.result

    assert isinstance(result, Result)
    assert result.model == regression.model
    assert result.cross_val_mean > 0
    assert result.cross_val_std > 0
    assert 'r2' == result.metric
    assert 'LinearRegression' == result.model_name
    assert 2 == len(result.cross_val_scores)


def test_regression_model_returns_a_result(classifier):
    result = classifier.result
    assert isinstance(result, Result)
    assert result.model == classifier.model
    assert result.cross_val_mean > 0
    assert result.cross_val_std > 0
    assert 'accuracy' == result.metric
    assert 'LogisticRegression' == result.model_name
    assert 2 == len(result.cross_val_scores)


def test_regression_model_can_be_saved(classifier, tmpdir, base):
    path = tmpdir.join('test.pkl')
    classifier.test_model()
    classifier.save_model(str(path))
    assert os.path.exists(path)
    loaded_model = base.load_model(str(path))
    assert loaded_model.model.get_params() == classifier.model.get_params()


def test_result_equality_operators():
    first_result = Result(model=None, model_name='test', cross_val_mean=.7, cross_val_std=.2)
    second_result = Result(model=None, model_name='test2', cross_val_mean=.5, cross_val_std=.2)

    assert first_result > second_result


def test_max_works_with_result():
    first_result = Result(model=None, model_name='test', cross_val_mean=.7, cross_val_std=.2)
    second_result = Result(model_name='test', model=None, cross_val_mean=.5, cross_val_std=.2)

    max_result = max([first_result, second_result])

    assert first_result is max_result


def test_make_prediction_errors_when_model_is_not_fitted(base):
    with pytest.raises(MLUtilsError, match="You haven't fitted the model"):
        model = base(LinearRegression())
        model.make_prediction(5)


def test_train_model_works_as_expected(regression):
    expected_x, expected_y = regression.get_training_data()
    regression.train_model()
    assert np.all(expected_x == regression.x)
    assert np.all(expected_y == regression.y)
