from sklearn.linear_model import LinearRegression

from ml_tooling import BaseClassModel
from ml_tooling.config import DefaultConfig


class TestConfig:

    def test_config_is_set_globally(self, pipeline_dummy_classifier, pipeline_linear):
        class TestModel(BaseClassModel):
            @classmethod
            def setup_model(cls):
                pass

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

    def test_can_change_config(self):
        class SomeModel(BaseClassModel):
            @classmethod
            def setup_model(cls):
                pass

            def get_training_data(self):
                pass

            def get_prediction_data(self, *args):
                pass

        SomeModel.reset_config()
        test_model = SomeModel(LinearRegression())
        assert 10 == test_model.config.CROSS_VALIDATION
        test_model.config.CROSS_VALIDATION = 2
        assert test_model.config.CROSS_VALIDATION == 2

    def test_config_repr_works(self):
        config = DefaultConfig()
        for key in ['VERBOSITY', 'CLASSIFIER_METRIC', 'REGRESSION_METRIC', 'CROSS_VALIDATION',
                    'STYLE_SHEET', 'N_JOBS', 'TEST_SIZE', 'RANDOM_STATE']:
            assert key in config.__repr__()

    def test_from_same_class_share_config(self, base, pipeline_logistic,
                                          pipeline_forest_classifier):
        log = base(pipeline_logistic)
        rf = base(pipeline_forest_classifier)
        assert log.config.CLASSIFIER_METRIC == 'accuracy'
        log.config.CLASSIFIER_METRIC = 'fowlkes_mallows_score'
        assert rf.config.CLASSIFIER_METRIC == 'fowlkes_mallows_score'

    def test_from_different_classes_do_not_share_config(self, base, base_second, pipeline_logistic,
                                                        pipeline_forest_classifier):
        log = base(pipeline_logistic)
        rf = base_second(pipeline_forest_classifier)
        assert log.config.CLASSIFIER_METRIC == 'accuracy'
        log.config.CLASSIFIER_METRIC = 'fowlkes_mallows_score'
        assert rf.config.CLASSIFIER_METRIC == 'accuracy'
