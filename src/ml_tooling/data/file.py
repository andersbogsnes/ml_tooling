import abc

from ml_tooling.data.base_data import Dataset
import pathlib

from ml_tooling.utils import Pathlike


class FileDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Baseclass for creating filebased Datasets. Subclass FileDataset and provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method. Filedataset takes a path
    as its initialization argument
    """

    def __init__(self, path: Pathlike):
        self.file_path = pathlib.Path(path)

    def __repr__(self):
        return f"<{self.__class__.__name__} - FileDataset>"
