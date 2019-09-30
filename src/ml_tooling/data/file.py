import abc
from typing import Union, Tuple

from ml_tooling.data.base_data import Dataset
import pathlib

from ml_tooling.utils import DataType


class FileDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Baseclass for creating filebased Datasets. Subclass FileDataset and provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method. Filedataset takes a path
    as its initialization argument
    """

    def __init__(self, path: Union[pathlib.Path, str]):
        self.file_path = pathlib.Path(path)

    @abc.abstractmethod
    def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
        pass

    @abc.abstractmethod
    def load_prediction_data(self, *args, **kwargs) -> DataType:
        pass
