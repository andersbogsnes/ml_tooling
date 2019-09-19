import abc
from typing import Union, Tuple

import pandas as pd
from ml_tooling.data.base_data import DataSet
from ml_tooling.utils import DataSetError
import pathlib

from ml_tooling.utils import DataType


class FileDataSet(DataSet):
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

    def save(self, overwrite=False, **kwargs):
        """
        Save dataset to path

        Parameters
        ----------
        overwrite: bool
            Whether or not to overwrite an existing file if exists
        kwargs:
            Passed to pandas to_* function

        Returns
        -------
        FileDataSet
        """

        if self.file_path.exists() and overwrite is False:
            raise FileExistsError(
                f"{self.file_path} already exists. "
                f"Set overwrite=True to overwrite existing file"
            )

        output_data = self.x.join(self.y)
        getattr(output_data, f"to_{self.file_type}")(self.file_path, **kwargs)
        return self
