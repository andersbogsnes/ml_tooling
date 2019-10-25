from ml_tooling.storage.base import Storage
from ml_tooling.utils import MLToolingError, find_src_dir

import joblib
from typing import List, Any
from pathlib import Path

from ml_tooling.utils import Pathlike, make_dir, Estimator


class FileStorage(Storage):
    """
    File Storage class for handling storage of estimators to the file system
    """

    def __init__(self, dir_path: Pathlike = None):
        self.dir_path = Path.cwd() if dir_path is None else Path(dir_path)
        if not self.dir_path.is_dir():
            raise MLToolingError(
                f"dir_path is {self.dir_path} which is not a directory"
            )

    def get_list(self) -> List[Path]:
        """
        Finds a list of estimator filenames in the FileStorage directory,
        if the path given is to a file, the directory in which the file resides
        is used to find the list.

        Example
        -------
        Find and return estimator paths in a given directory:
            my_estimators = FileStorage('path/to/dir').get_list()

        Returns
        -------
        List[Path]
            list of paths to files sorted by filename
        """
        return sorted(self.dir_path.glob("*.pkl"))

    def load(self, filename: str) -> Any:
        """
        Loads a joblib pickled estimator from given filepath and returns the unpickled object

        Parameters
        ----------

        filename: str
            filename of estimator pickle file

        Example
        -------
        We can load a saved pickled estimator from disk directly from FileStorage:

            storage = FileStorage('path/to/dir')
            my_estimator = storage.load('mymodel.pkl')

        We now have a trained estimator loaded.

        Returns
        -------
        Object
            The object loaded from disk
        """
        estimator_path = self.dir_path / filename
        return joblib.load(estimator_path)

    def save(self, estimator: Estimator, filepath: str, prod: bool = False) -> Path:
        """
        Save a joblib pickled estimator.

        Parameters
        ----------
        estimator: obj
            The estimator object

        filepath: str
            Path where to save file - relative to FileStorage

        prod: bool
            Whether or not to save in "production mode" -
            Production mode saves to /src/<projectname>/ regardless of what FileStorage
            was instantiated with

        Example
        -------
        To save your trained estimator, use the FileStorage context manager.

            storage = FileStorage('/path/to/save/dir/')
            file_path = storage.save(estimator, 'filename')

        We now have saved an estimator to a pickle file.

        Returns
        -------
        Path
            Path to the saved object
        """

        if prod:
            file_path = find_src_dir() / filepath
        else:
            file_path = make_dir(self.dir_path) / filepath

        joblib.dump(estimator, file_path)
        return file_path
