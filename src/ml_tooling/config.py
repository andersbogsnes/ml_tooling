import pathlib

from ml_tooling.storage import FileStorage

BASE_PATH = pathlib.Path(__file__).parent
CWD = pathlib.Path.cwd()

MPL_STYLESHEET = str(BASE_PATH.joinpath("almbrand.mplstyle"))
RUN_DIR = CWD.joinpath("runs")
ESTIMATOR_DIR = CWD.joinpath("models")


class DefaultConfig:
    """
    Configuration for a given BaseClass. Configs propagate through each instance
    """

    default_config = {
        "VERBOSITY": 0,
        "CLASSIFIER_METRIC": "accuracy",
        "REGRESSION_METRIC": "r2",
        "CROSS_VALIDATION": 10,
        "N_JOBS": -1,
        "RANDOM_STATE": 42,
        "TRAIN_TEST_SHUFFLE": True,
        "TEST_SIZE": 0.25,
    }

    def __init__(self):
        self._set_config()
        self.LOG = False
        self.RUN_DIR = RUN_DIR
        self.ESTIMATOR_DIR = ESTIMATOR_DIR

    def _set_config(self):
        self.VERBOSITY = self.default_config["VERBOSITY"]
        self.CLASSIFIER_METRIC = self.default_config["CLASSIFIER_METRIC"]
        self.REGRESSION_METRIC = self.default_config["REGRESSION_METRIC"]
        self.CROSS_VALIDATION = self.default_config["CROSS_VALIDATION"]
        self.N_JOBS = self.default_config["N_JOBS"]
        self.RANDOM_STATE = self.default_config["RANDOM_STATE"]
        self.TRAIN_TEST_SHUFFLE = self.default_config["TRAIN_TEST_SHUFFLE"]
        self.TEST_SIZE = self.default_config["TEST_SIZE"]

    @property
    def default_storage(self):
        return FileStorage(self.ESTIMATOR_DIR)

    def reset_config(self):
        self._set_config()

    def __repr__(self):
        attrs = "\n".join(
            [
                f"{attr}: {value}"
                for attr, value in self.__dict__.items()
                if "__" not in attr
            ]
        )
        return f"<Config: \n{attrs}\n>"


config = DefaultConfig()
