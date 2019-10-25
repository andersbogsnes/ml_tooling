from abc import ABCMeta, abstractmethod
from typing import List

from ml_tooling.utils import Pathlike, Estimator


class Storage(metaclass=ABCMeta):
    """
    Base class for Storage classes
    """

    @abstractmethod
    def load(self, filename: str) -> Estimator:
        """
        Abstract method to be implemented by the user.
        Defines method used to load data from the storage type

        Returns
        -------
        Estimator
            Returns the unpickled object
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, estimator: Estimator, filename: str, prod: bool = False) -> Pathlike:
        """
        Abstract method to be implemented by the user.
        Defines method used to save data from the storage type

        Returns
        -------
        Pathlike
            Path to where the pickled object is saved
        """
        raise NotImplementedError

    @abstractmethod
    def get_list(self) -> List[Pathlike]:
        """
        Abstract method to be implemented by the user.
        Defines method used to show which objects have been saved

        Returns
        -------
        List[Path]
            Paths to each of the estimators sorted lexically
        """
        raise NotImplementedError
