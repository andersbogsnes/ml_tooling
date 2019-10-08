from ml_tooling.storage.base import Storage
from ml_tooling.config import ConfigGetter
from ml_tooling.utils import MLToolingError

from pathlib import Path

try:
    from artifactory import ArtifactoryPath
except ImportError:
    raise MLToolingError("Artifactory not installed - run pip install dohq-artifactory")


class ArtifactoryStorage(Storage):
    
    def __init__(self, repo_path=None, auth: Tuple[str, str] = None):
        self.auth = auth
        self.repo_path = repo_path

    def load(self, filename):
        file_path = f'{self.repo_path}{filename}'
        artifactory_path = ArtifactoryPath(file_path, auth=self.auth)
        with artifactory_path.open() as f:
            return f.read()

    def save(self, filepath, env: StorageEnvironment = StorageEnvironment.DEV
             ):
        environment_path = f'{self.repo_path}/{env.name}/'
        artifactory_path = ArtifactoryPath(environment_path, auth=self.auth)
        artifactory_path.mkdir(exist_ok=True)
        return artifactory_path.deploy_file(filepath)

    def get_list(self):
        # get estimators for this dataset? dataset + model?
        pass