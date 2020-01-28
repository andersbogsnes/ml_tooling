import abc
import logging
from typing import Tuple

import pandas as pd
from ml_tooling.data.base_data import Dataset
import pathlib

from ml_tooling.utils import Pathlike, DatasetError, DataType

logger = logging.getLogger("ml_tooling")


class FileDataset(Dataset, metaclass=abc.ABCMeta):
    """
    An Abstract Base Class for use in creating Filebased Datasets.
    This class is intended to be subclassed and must provide a :meth:`load_training_data`
    and :meth:`load_prediction_data` method.

    FileDataset takes a path as its initialization argument, pointing to a file which must be a
    filetype supported by Pandas, such as csv, parquet etc. The extension determines the
    pandas method used to read and write the data


    Methods
    -------
    load_prediction_data(idx, conn)
        Used to load prediction data for a given idx - returns features

    load_training_data(conn)
        Used to load the full training dataset - returns features and targets

    """

    def __init__(self, path: Pathlike):
        """
        Instantiates a Filedataset pointing at a given path.

        Parameters
        ----------
        path: Pathlike
            Path to location of file
        """
        self.file_path = pathlib.Path(path)

        if self.file_path.suffix == "":
            raise DatasetError(f"{self.file_path} must point to a file")

        self.extension = self.file_path.suffix[1:]

    @abc.abstractmethod
    def load_training_data(
        self, conn, *args, **kwargs
    ) -> Tuple[pd.DataFrame, DataType]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_prediction_data(self, conn, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _dump_data(self, **kwargs) -> pd.DataFrame:
        """
        Reads the underlying file and returns a DataFrame

        Parameters
        ----------
        kwargs: dict
            Kwargs are passed to the relevant :meth:`pd.read_*` method for the given extension

        Returns
        -------
        pd.DataFrame

        """
        return getattr(pd, f"read_{self.extension}")(self.file_path, **kwargs)

    def _load_data(self, data: pd.DataFrame, **kwargs):
        """
        Writes input data to the underlying file

        Parameters
        ----------
        data: pd.DataFrame
            Input data to write to file
        kwargs: dict
            Passed to the relevant :meth:`pd.DataFrame.to_*` method

        """
        index = kwargs.pop("index", False)
        getattr(data, f"to_{self.extension}")(self.file_path, index=index, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__} - FileDataset>"
