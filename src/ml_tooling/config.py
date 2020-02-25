import pathlib
import configparser
from ml_tooling.storage import FileStorage

PACKAGE_PATH = pathlib.Path(__file__).parent
MPL_STYLESHEET = str(PACKAGE_PATH.joinpath("almbrand.mplstyle"))

config_search_path = [
    pathlib.Path().cwd().parent.joinpath("ml_tooling.ini"),
    pathlib.Path().cwd().joinpath("ml_tooling.ini"),
]

parser = configparser.ConfigParser(
    defaults={
        "verbosity": 0,
        "classifier_metric": "accuracy",
        "regression_metric": "r2",
        "cross_validation": "10",
        "n_jobs": -1,
        "random_state": 42,
        "run_dir": "runs",
        "estimator_dir": "models",
    },
    default_section="ml_tooling",
)


class DefaultConfig:
    """
    Configuration for a given BaseClass. Configs propagate through each instance
    """

    def __init__(self):
        parser.read(config_search_path)
        config = parser["ml_tooling"]

        self.VERBOSITY = config.getint("verbosity")
        self.CLASSIFIER_METRIC = config["classifier_metric"]
        self.REGRESSION_METRIC = config["regression_metric"]
        self.CROSS_VALIDATION = config.getint("cross_validation")
        self.N_JOBS = config.getint("n_jobs")
        self.RANDOM_STATE = config.getint("random_state")
        self.RUN_DIR = self.config_file.parent / config["run_dir"]
        self.ESTIMATOR_DIR = self.config_file.parent / config["estimator_dir"]
        self.LOG = False
        self.STYLE_SHEET = MPL_STYLESHEET

    @property
    def config_file(self):
        for path in config_search_path:
            if path.exists():
                return path
            return pathlib.Path.cwd()

    @property
    def default_storage(self):
        return FileStorage(self.ESTIMATOR_DIR)

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
