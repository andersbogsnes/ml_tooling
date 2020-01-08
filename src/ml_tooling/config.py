import pathlib

BASE_PATH = pathlib.Path(__file__).parent
CWD = pathlib.Path.cwd()

MPL_STYLESHEET = str(BASE_PATH.joinpath("almbrand.mplstyle"))
RUN_DIR = CWD.joinpath("runs")
ESTIMATOR_DIR = CWD.joinpath("models")


class DefaultConfig:
    """
    Configuration for a given BaseClass. Configs propagate through each instance
    """

    def __init__(self):
        self.VERBOSITY = 0
        self.CLASSIFIER_METRIC = "accuracy"
        self.REGRESSION_METRIC = "r2"
        self.CROSS_VALIDATION = 10
        self.STYLE_SHEET = MPL_STYLESHEET
        self.N_JOBS = -1
        self.RANDOM_STATE = 42
        self.RUN_DIR = RUN_DIR
        self.ESTIMATOR_DIR = ESTIMATOR_DIR
        self.LOG = False

    def __repr__(self):
        attrs = "\n".join(
            [
                f"{attr}: {value}"
                for attr, value in self.__dict__.items()
                if "__" not in attr
            ]
        )
        return f"<Config: \n{attrs}\n>"


class ConfigGetter:
    """
    Give each class that inherits from Model an individual config attribute
    without relying on the user to overriding the config when they define their class.
    """

    def __get__(self, obj, cls):
        if cls._config is None:
            cls._config = DefaultConfig()
        return cls._config
