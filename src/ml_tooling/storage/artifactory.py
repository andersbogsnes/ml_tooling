from ml_tooling.storage import Storage, StorageEnvironment
from ml_tooling.utils import MLToolingError

import os
import joblib
from tempfile import TemporaryDirectory
from typing import Tuple, Union, Optional, List
from pathlib import Path
from sklearn.base import BaseEstimator

try:
    from artifactory import ArtifactoryPath
except ImportError:
    raise MLToolingError("Artifactory not installed - run pip install dohq-artifactory")


class ArtifactoryStorage(Storage):
    """
    Artifactory Storage class
    """

    def __init__(
        self,
        repo_path: Union[Path, str] = None,
        apikey: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None,
    ):
        self.repo_path = repo_path
        self.auth = auth
        self.apikey = apikey

    def get_list(self) -> List[ArtifactoryPath]:
        """
        Finds a list of estimator filenames in the ArtifactoryStorage repo,
        if the path given is for a file, the directory in which the file resides
        is used to find the list.

        Parameters
        ----------
        None

        Example
        -------
        Find and return estimator paths in a given directory:
            my_estimators = ArtifactoryStorage('path/to/dir').get_list()

        Returns
        -------
        List[ArtifactoryPath]
            list of paths to files
        """
        return sorted(ArtifactoryPath(self.repo_path).glob("*/*.pkl"))

    def load(self, filename: Union[str, Path, ArtifactoryPath]) -> BaseEstimator:
        """
        Loads a pickled estimator from given filepath and returns the estimator

        Parameters
        ----------
        file_path: str, Path, ArtifactoryPath
            Path to estimator pickle file

        Example
        -------
        We can load a saved pickled estimator from disk directly from FileStorage:
            storage = ArtifactoryStorage()
            my_estimator = storage.load('http://yourdomain/path/to/estimator')

        We now have a trained estimator loaded.

        Returns
        -------
        Object
            estimator unpickled object
        """
        if self.repo_path is None:
            filepath = filename
        else:
            filepath = f"{self.repo_path}{filename}"

        artifactory_path = ArtifactoryPath(filepath, auth=self.auth, apikey=self.apikey)
        with artifactory_path.open() as f:
            with TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir).joinpath("temp.pkl")
                with open(filepath, "wb") as out:
                    out.write(f.read())
                return joblib.load(filepath)

    def save(
        self,
        estimator: BaseEstimator,
        filepath: Union[Path, str],
        env: StorageEnvironment = StorageEnvironment.dev,
    ) -> ArtifactoryPath:
        """
        Save a pickled estimator to artifactory.

        Parameters
        ----------
        estimator: BaseEstimator
            The estimator object

        Example
        -------
        To save your trained estimator:
            storage = ArtifactoryStorage('http://yourdomain/path/to/save/dir/')
            artyfactory_path = storage.save(estimator)

        We now have saved an estimator to a pickle file.

        Returns
        -------
        BaseEstimator
            artyfactory_path: artifactory file path
        """
        path_with_env = f"{self.repo_path}/{env.name}/{os.path.basename(filepath)}"
        artifactory_path = ArtifactoryPath(
            path_with_env, auth=self.auth, apikey=self.apikey
        )
        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir).joinpath("temp.pkl")
            joblib.dump(estimator, filepath)
            artifactory_path.deploy_file(filepath)
        return artifactory_path
