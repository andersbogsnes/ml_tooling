import pathlib
import configparser
from typing import List, Optional
import attr

from ml_tooling.storage import FileStorage

PACKAGE_PATH = pathlib.Path(__file__).parent
MPL_STYLESHEET = str(PACKAGE_PATH.joinpath("almbrand.mplstyle"))


class ConfigLoader:
    def __init__(
        self,
        search_path: Optional[List[pathlib.Path]] = None,
        config_filename: str = "ml_tooling.ini",
    ):
        self._parser = configparser.ConfigParser(
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
        search_path = (
            [pathlib.Path().cwd().parent, pathlib.Path().cwd()]
            if search_path is None
            else search_path
        )
        self._config_filename = config_filename
        self._search_path = [
            path.joinpath(self._config_filename) for path in search_path
        ]
        self._parser.read(self._search_path)
        self.default_config = self._parser["ml_tooling"]

    @property
    def config_location(self) -> pathlib.Path:
        for path in self._search_path:
            if path.joinpath(self._config_filename).exists():
                return path
            return pathlib.Path.cwd()


@attr.s(auto_attribs=True)
class DefaultConfig:
    """
    Configuration for a given BaseClass. Configs propagate through each instance
    """

    VERBOSITY: int
    CLASSIFIER_METRIC: str
    REGRESSION_METRIC: str
    CROSS_VALIDATION: int
    N_JOBS: int
    RANDOM_STATE: int
    RUN_DIR: pathlib.Path
    ESTIMATOR_DIR: pathlib.Path
    STYLE_SHEET: str
    LOG: bool = False

    @classmethod
    def from_configloader(cls, loader: ConfigLoader):
        config = loader.default_config
        return cls(
            VERBOSITY=config.getint("verbosity"),
            CLASSIFIER_METRIC=config["classifier_metric"],
            REGRESSION_METRIC=config["regression_metric"],
            CROSS_VALIDATION=config.getint("cross_validation"),
            N_JOBS=config.getint("n_jobs"),
            RANDOM_STATE=config.getint("random_state"),
            RUN_DIR=loader.config_location / config["run_dir"],
            ESTIMATOR_DIR=loader.config_location / config["estimator_dir"],
            LOG=False,
            STYLE_SHEET=MPL_STYLESHEET,
        )

    @property
    def default_storage(self):
        return FileStorage(self.ESTIMATOR_DIR)


class ConfigGetter:
    """
    Give each class that inherits from Model an individual config attribute
    without relying on the user to overriding the config when they define their class.
    """

    def __get__(self, obj, cls):
        if cls._config is None:
            loader = ConfigLoader()
            cls._config = DefaultConfig.from_configloader(loader=loader)
        return cls._config
