import abc
from typing import Union

from ml_tooling.data.base_data import Dataset
import pathlib


class FileDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Baseclass for creating filebased Datasets. Subclass FileDataset and provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method. Filedataset takes a path
    as its initialization argument
    """

    def __init__(self, path: Union[pathlib.Path, str]):
        self.file_path = pathlib.Path(path)
