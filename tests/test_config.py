import pytest

from ml_tooling import Model
from ml_tooling.config import DefaultConfig, ConfigLoader
import configparser


class TestConfig:
    @pytest.fixture()
    def test_model(self):
        class TestModel(Model):
            def get_prediction_data(self, *args):
                pass

            def get_training_data(self):
                pass

        TestModel.reset_config()
        return TestModel

    def test_instance_of_model_inherits_top_level(
        self, test_model, pipeline_dummy_classifier
    ):
        model = test_model(pipeline_dummy_classifier)
        assert model.config.N_JOBS == -1

    def test_changing_model_config_changes_instance_config(
        self, test_model, pipeline_dummy_classifier
    ):
        model = test_model(pipeline_dummy_classifier)
        test_model.config.N_JOBS = 1
        assert model.config.N_JOBS == 1

    def test_changing_model_config_before_instantiating_model_changes_instance_config(
        self, test_model, pipeline_dummy_classifier
    ):
        test_model.config.N_JOBS = 1
        model = test_model(pipeline_dummy_classifier)
        assert model.config.N_JOBS == 1

    def test_config_repr_works(self):
        loader = ConfigLoader()
        config = DefaultConfig.from_configloader(loader)
        for key in [
            "VERBOSITY",
            "CLASSIFIER_METRIC",
            "REGRESSION_METRIC",
            "CROSS_VALIDATION",
            "STYLE_SHEET",
            "N_JOBS",
            "RANDOM_STATE",
        ]:
            assert key in config.__repr__()

    def test_from_same_class_share_config(
        self, base, pipeline_logistic, pipeline_forest_classifier
    ):
        log = base(pipeline_logistic)
        rf = base(pipeline_forest_classifier)
        assert log.config.CLASSIFIER_METRIC == "accuracy"
        log.config.CLASSIFIER_METRIC = "fowlkes_mallows_score"
        assert rf.config.CLASSIFIER_METRIC == "fowlkes_mallows_score"

    def test_from_different_classes_do_not_share_config(
        self, base, pipeline_logistic, pipeline_forest_classifier
    ):
        class NoModel(Model):
            def get_prediction_data(self, idx):
                pass

            def get_training_data(self):
                pass

        log = base(pipeline_logistic)
        rf = NoModel(pipeline_forest_classifier)
        assert log.config.CLASSIFIER_METRIC == "accuracy"
        log.config.CLASSIFIER_METRIC = "fowlkes_mallows_score"
        assert rf.config.CLASSIFIER_METRIC == "accuracy"
        assert log.config.CLASSIFIER_METRIC == "fowlkes_mallows_score"

    def test_config_default_storage_points_to_cwd(self, base, tmp_path):
        base.config.ESTIMATOR_DIR = tmp_path
        result = base.config.default_storage

        assert result.dir_path == tmp_path

    def test_using_config_file_changes_config(self, tmp_path):
        parser = configparser.ConfigParser()
        parser["ml_tooling"] = {"classifier_metric": "average_precision"}
        tmp_path
