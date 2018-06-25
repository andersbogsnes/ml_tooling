import pytest

from ml_utils.baseclass._baseclass import Result, MLUtilsError
from ml_utils import BaseClassModel
from sklearn.linear_model import LinearRegression


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
    assert 'roc_auc' == result.metric
    assert 'LogisticRegression' == result.model_name
    assert 2 == len(result.cross_val_scores)
