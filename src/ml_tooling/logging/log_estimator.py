import pathlib
from datetime import datetime
from typing import Optional

import yaml

from ml_tooling.metrics import Metrics
from ml_tooling.utils import get_git_hash, make_dir, Pathlike
import attr


@attr.s(auto_attribs=True)
class Log:
    """
    Log object containing all relevant data for generating a logfile describing a given result

    Can be instantiated directly from a result using :meth:`from_result`, passing the estimator
    path if available

    Save a yaml file to disk using :meth:`save_log`
    """

    name: str
    metrics: Metrics = attr.ib(repr=False)
    estimator: dict = attr.ib(repr=False)
    estimator_path: Optional[pathlib.Path] = attr.ib(default=None)
    output_path: Optional[pathlib.Path] = attr.ib(default=None)

    @classmethod
    def from_result(cls, result, estimator_path: Optional[pathlib.Path] = None):
        return cls(
            name=f"{result.data.class_name}_{result.model.estimator_name}",
            metrics=result.metrics,
            estimator=result.model.to_dict(),
            estimator_path=estimator_path,
        )

    def _generate_output_path(self, save_dir: pathlib.Path) -> pathlib.Path:
        """
        Generates a valid filename, appending an incrementing counter if
        file with same name already exists

        Parameters
        ----------
        save_dir: pathlib.Path
            Directory where logs are to be saved

        Returns
        -------
        pathlib.Path
            Path to output_file

        """
        save_dir = make_dir(save_dir)
        now = datetime.now()
        iteration = 0
        output_file = f'{self.name}_{now.strftime("%H%M%S")}_{iteration}.yaml'
        output_path = save_dir.joinpath(output_file)

        while output_path.exists():
            output_file = f'{self.name}_{now.strftime("%H%M%S")}_{iteration}.yaml'
            output_path = save_dir.joinpath(output_file)
            iteration += 1

        return output_path

    def dump(self) -> dict:
        """
        Creates a dictionary log of the model, including the serialized model,
        path to the saved estimator and scores

        Returns
        -------
        dict
            Dictionary containing:
            * model_name
            * created_time
            * versions
                - ml_tooling -> version
                - sklearn -> version
                - pandas -> version
            * git_hash
            * metrics
                - metric -> score
            * estimator
                - name -> name of pipeline step
                - module -> name of module
                - classname -> name of class
                - params -> dict of params
            * estimator_path
        """
        from ml_tooling import __version__ as ml_tools_version
        from sklearn import __version__ as sklearn_version
        from pandas import __version__ as pandas_version

        versions = {
            "ml_tooling": ml_tools_version,
            "sklearn": sklearn_version,
            "pandas": pandas_version,
        }

        data = {
            "model_name": self.name,
            "created_time": datetime.now(),
            "versions": versions,
            "git_hash": get_git_hash(),
            "metrics": self.metrics.to_dict(),
            "estimator": self.estimator,
            "estimator_path": str(self.estimator_path) if self.estimator_path else None,
        }
        return data

    def save_log(self, save_dir: Pathlike) -> "Log":
        """
        Saves a log to a given directory

        Parameters
        ----------
        save_dir: pathlib.Path
            Directory to save logfile in

        Returns
        -------
        pathlib.Path
            Path where logfile was saved
        """
        save_dir = pathlib.Path(save_dir)
        output_path: pathlib.Path = self._generate_output_path(save_dir)
        log = self.dump()

        with output_path.open(mode="w") as f:
            yaml.safe_dump(log, f, default_flow_style=False, allow_unicode=True)
        self.output_path = output_path
        return self
