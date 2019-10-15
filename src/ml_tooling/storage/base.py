from abc import ABCMeta, abstractmethod


class Storage(metaclass=ABCMeta):
    """
    Base class for Storage classes
    """

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
