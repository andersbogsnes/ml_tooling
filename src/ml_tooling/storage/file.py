from ml_tooling.storage.base import Storage

import joblib
from typing import List, Any, Union
from pathlib import Path
from sklearn.base import BaseEstimator


class FileStorage(Storage):
    """
    File Storage class for handling storage of estimators to the file system
    """

    def __init__(self, dir_path=None):
        self.dir_path = Path.cwd() if dir_path is None else Path(dir_path)

    def get_list(self) -> List[Path]:
        """
        Finds a list of estimator filenames in the FileStorage directory,
        if the path given is to a file, the directory in which the file resides
        is used to find the list.

        Parameters
        ----------
        None

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

    def load(self, file_path: Union[Path, str]) -> Any:
        """
        Loads a joblib pickled estimator from given filepath and returns a Model
        instatiated with the estimator.

        Parameters
        ----------
        file_path: str
            Path to estimator pickle file

        Example
        -------
        We can load a saved pickled estimator from disk directly from FileStorage:
            storage = FileStorage()
            my_estimator = storage.load('path/to/estimator')

        We now have a trained estimator loaded.

        Returns
        -------
        Object
            any python object loaded from disk
        """
        estimator_path = Path(file_path)
        return joblib.load(estimator_path)

    def save(self, estimator: BaseEstimator, filename: str) -> Path:
        """
        Save a joblib pickled estimator.

        Parameters
        ----------
        estimator: obj
            The estimator object

        Example
        -------
        To save your trained estimator, use the FileStorage context manager.
            storage = FileStorage('/path/to/save/dir/')
            file_path = storage.save(estimator, 'filename')

        We now have saved an estimator to a pickle file.

        Returns
        -------
        Path
            file_path: pathlib Path
        """
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)
        file_path = self.dir_path.joinpath(filename)
        joblib.dump(estimator, file_path)
        return file_path
