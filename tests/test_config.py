from ml_tooling.config import DefaultConfig


class TestConfig:
    def test_config_repr_works(self):
        config = DefaultConfig()
        for key in [
            "VERBOSITY",
            "CLASSIFIER_METRIC",
            "REGRESSION_METRIC",
            "CROSS_VALIDATION",
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
        log.config.reset_config()

    def test_config_default_storage_points_to_cwd(self, base, tmp_path):
        base.config.ESTIMATOR_DIR = tmp_path
        result = base.config.default_storage

        assert result.dir_path == tmp_path
