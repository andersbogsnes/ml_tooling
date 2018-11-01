from sklearn.linear_model import LinearRegression

from ml_tooling import BaseClassModel
from ml_tooling.config import DefaultConfig


def test_config_is_set_globally(pipeline_dummy_classifier, pipeline_linear):
    class TestModel(BaseClassModel):
        def get_prediction_data(self, *args):
            pass

        def get_training_data(self):
            pass

    TestModel.reset_config()

    assert TestModel.config.N_JOBS == -1

    model = TestModel(pipeline_dummy_classifier)
    assert model.config.N_JOBS == -1

    TestModel.config.N_JOBS = 1
    assert TestModel.config.N_JOBS == 1
    assert model.config.N_JOBS == 1

    new_model = TestModel(pipeline_dummy_classifier)
    assert new_model.config.N_JOBS == 1


def test_can_change_config():
    class SomeModel(BaseClassModel):
        def get_training_data(self):
            pass

        def get_prediction_data(self, *args):
            pass

    SomeModel.reset_config()
    test_model = SomeModel(LinearRegression())
    assert 10 == test_model.config.CROSS_VALIDATION
    test_model.config.CROSS_VALIDATION = 2
    assert test_model.config.CROSS_VALIDATION == 2


def test_config_repr_works():
    config = DefaultConfig()
    for key in ['VERBOSITY', 'CLASSIFIER_METRIC', 'REGRESSION_METRIC', 'CROSS_VALIDATION',
                'STYLE_SHEET', 'N_JOBS', 'TRAIN_SIZE', 'RANDOM_STATE']:
        assert key in config.__repr__()
