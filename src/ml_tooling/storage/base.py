
from abc import ABCMeta, abstractmethod, abstractclassmethod
from enum import Enum

class StorageEnvironment(Enum):
    DEV = 1
    TEST = 2
    PROD = 3

class Storage(metaclass=ABCMeta):
    @abstractmethod
    def load(self, file_path):
        """
        Abstract method to be implemented by the user.
        Defines method used to load data from the storage type

        Returns
        -------
        Model
            Instance of Model with a saved estimator
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, estimator_file, filename):
        """
        Abstract method to be implemented by the user.
        Defines method used to save data from the storage type

        Returns
        -------
        Model
            Instance of Model with a saved estimator
        """
        raise NotImplementedError
