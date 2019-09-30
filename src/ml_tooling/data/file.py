import abc
from typing import Union, Tuple
import pathlib

from ml_tooling.data.base_data import DataSet
from ml_tooling.utils import DataSetError, DataType

import pandas as pd


class FileDataSet(DataSet, metaclass=abc.ABCMeta):
    def __init__(self, path: Union[pathlib.Path, str]):
        self.file_path = pathlib.Path(path)
        self.file_type = self.file_path.suffix[1:]

        if not hasattr(pd.DataFrame, f"to_{self.file_type}"):
            raise DataSetError(f"{self.file_type} not supported")

    @abc.abstractmethod
    def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
        pass

    @abc.abstractmethod
    def load_prediction_data(self, *args, **kwargs) -> DataType:
        pass
