from ml_tooling.storage import Storage
from ml_tooling.utils import MLToolingError

import joblib
from tempfile import TemporaryDirectory
from typing import Tuple, Union, Optional, List, Text
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

try:
    from artifactory import ArtifactoryPath
except ImportError:
    raise MLToolingError(
        "Artifactory not installed - run pip install dohq-artifactory"
    ) from None


class ArtifactoryStorage(Storage):
    """
    Artifactory Storage class for handling storage of estimators to JFrog artifactory

    Example
    -------
    Instantiate this class with a url and path to the repo like so:
        storage = ArtifactoryStorage('http://artifactory.com', pathlib.Path('/path/to/artifact'))
    """

    def __init__(
        self,
        artifactory_url: Text,
        repo_path: Union[Path, str],
        apikey: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None,
    ):
        self.artifactory_url = artifactory_url
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
            my_estimators = ArtifactoryStorage('http://artifactory.com', 'path/to/repo').get_list()

        Returns
        -------
        List[ArtifactoryPath]
            list of paths to files sorted by filename
        """
        return sorted(
            ArtifactoryPath(f"{self.artifactory_url}{self.repo_path}").glob("*/*.pkl")
        )

    def load(
        self, filename: Union[str, Path, ArtifactoryPath]
    ) -> Union[BaseEstimator, Pipeline]:
        """
        Loads a pickled estimator from given filepath and returns the estimator

        Parameters
        ----------
        file_path: str, Path, ArtifactoryPath
            Path to estimator pickle file

        Example
        -------
        We can load a saved pickled estimator from disk directly from FileStorage:
            storage = ArtifactoryStorage('http://artifactory.com', 'path/to/repo')
            my_estimator = storage.load('estimatorfile')

        We now have a trained estimator loaded.

        Returns
        -------
        Object
            estimator unpickled object
        """
        if isinstance(filename, type(ArtifactoryPath)):
            filepath = filename
        else:
            filepath = f"{self.artifactory_url}/{self.repo_path}/{filename}".replace(
                "//", "/"
            )

        artifactory_path = ArtifactoryPath(filepath, auth=self.auth, apikey=self.apikey)
        with artifactory_path.open() as f:
            with TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir).joinpath("temp.pkl")
                with open(filepath, "wb") as out:
                    out.write(f.read())
                return joblib.load(filepath)

    def save(
        self,
        estimator: Union[BaseEstimator, Pipeline],
        filename: str,
        prod: bool = False,
    ) -> ArtifactoryPath:
        """
        Save a pickled estimator to artifactory.

        Parameters
        ----------
        estimator: Union[BaseEstimator, Pipeline]
            The estimator object
        filename: str
            Filename for the saved estimator
        prod: bool
            Production variable, set to True if saving a production-ready estimator

        Example
        -------
        To save your trained estimator:
            storage = ArtifactoryStorage('http://artifactory.com', 'path/to/repo')
            artifactory_path = storage.save(estimator)
        For production ready models set the prod parameter to True
            artifactory_path = storage.save(estimator, prod=True)

        We now have saved an estimator to a pickle file.

        Returns
        -------
        ArtifactoryPath
            artifactory_path: artifactory file path
        """
        env_name = "prod" if prod else "dev"
        repo = f"{self.artifactory_url}{self.repo_path}"
        path_with_env = f"{repo}/{env_name}/{filename}".replace("//", "/")
        artifactory_path = ArtifactoryPath(
            path_with_env, auth=self.auth, apikey=self.apikey
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir).joinpath("temp.pkl")
            joblib.dump(estimator, file_path)
            artifactory_path.deploy_file(file_path)
        return artifactory_path
