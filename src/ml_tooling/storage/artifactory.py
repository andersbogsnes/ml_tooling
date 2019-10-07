from ml_tooling.storage.base import Storage
from ml_tooling.config import ConfigGetter
from ml_tooling.utils import MLToolingError

from pathlib import Path

try:
    from artifactory import ArtifactoryPath
except ImportError:
    raise MLToolingError("Artifactory not installed - run pip install dohq-artifactory")


class ArtifactoryStorage(Storage):
    
    def __init__(self, repo_path=None, auth=None):
        self.auth = auth
        self.repo_path = Path(repo_path)
        self.filename = None

    def load(self):
        file_path = self.repo_path.joinpath(self.filename)
        artifactory_path = ArtifactoryPath(
            file_path,
            auth=self.auth
        )
        
        with artifactory_path.open() as f:
            return f.read()

    def save(self, filename):
        file_path = self.repo_path + filename
        artifactory_path = ArtifactoryPath(file_path, auth=self.auth)
        return artifactory_path.deploy_file(filename)

    def get_list(self):
        # get estimators for this dataset? dataset + model?
        pass