from io import BytesIO

from ml_tooling.storage import Storage
from ml_tooling.utils import Estimator, MLToolingError

import joblib
from tempfile import TemporaryDirectory
from typing import Tuple, Optional, List
from pathlib import Path

_has_artifactory = True

try:
    from artifactory import ArtifactoryPath
except ImportError:
    _has_artifactory = False


def generate_url(
    artifactory_url: str, repo: str, apikey=None, auth=None
) -> "ArtifactoryPath":
    if not artifactory_url.endswith("artifactory"):
        artifactory_url = f"{artifactory_url}/artifactory"

    return ArtifactoryPath(f"{artifactory_url}/{repo}", apikey=apikey, auth=auth)


class ArtifactoryStorage(Storage):
    """
    Artifactory Storage class for handling storage of estimators to JFrog artifactory

    Example
    -------
    Instantiate this class with a url and path to the repo like so:

        storage = ArtifactoryStorage('http://artifactory.com','/path/to/artifact')
    """

    def __init__(
        self,
        artifactory_url: str,
        repo: str,
        apikey: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None,
    ):

        if not _has_artifactory:
            raise MLToolingError(
                "Artifactory not installed - run pip install dohq-artifactory"
            )

        self.artifactory_path: ArtifactoryPath = generate_url(
            artifactory_url, repo, apikey, auth
        )

    def get_list(self) -> List["ArtifactoryPath"]:
        """
        Finds a list of estimator filenames in the ArtifactoryStorage repo,
        if the path given is for a file, the directory in which the file resides
        is used to find the list.

        Example
        -------
        Find and return estimator paths in a given directory:
            my_estimators = ArtifactoryStorage('http://artifactory.com', 'path/to/repo').get_list()

        Returns
        -------
        List[ArtifactoryPath]
            list of paths to files sorted by filename
        """

        return sorted(self.artifactory_path.glob("*/*.pkl"))

    def load(self, filename: str) -> Estimator:
        """
        Loads a pickled estimator from given filepath and returns the estimator

        Parameters
        ----------
        filename: str
            Path to estimator pickle file

        Example
        -------
        We can load a saved pickled estimator from disk directly from Artifactory:

            storage = ArtifactoryStorage('http://artifactory.com', 'path/to/repo')
            my_estimator = storage.load('estimatorfile')

        We now have a trained estimator loaded.

        Returns
        -------
        Object
            estimator unpickled object
        """

        filename = Path(filename).name

        artifactory_path = self.artifactory_path / filename
        with artifactory_path.open() as f:
            by = BytesIO()
            by.write(f.read())
            by.seek(0)
            return joblib.load(by)

    def save(
        self, estimator: Estimator, filename: str, prod: bool = False
    ) -> "ArtifactoryPath":
        """
        Save a pickled estimator to artifactory.

        Parameters
        ----------
        estimator: Estimator
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
            File path to stored estimator
        """

        if prod:
            raise NotImplementedError(
                "Artifactory Storage doesn't currently implement production storage. "
                "Use FileStorage instead"
            )

        env_path = "prod" if prod else "dev"

        artifactory_path = self.artifactory_path / env_path

        artifactory_path.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir).joinpath(filename)
            joblib.dump(estimator, file_path)
            artifactory_path.deploy_file(file_path)
        return artifactory_path
