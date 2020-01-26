import abc
import logging

import pandas as pd
from ml_tooling.data.base_data import Dataset
import pathlib

from ml_tooling.utils import Pathlike

logger = logging.getLogger("ml_tooling")


class FileDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Baseclass for creating filebased Datasets. Subclass FileDataset and provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method. Filedataset takes a path
    as its initialization argument which must be a filetype supported by Pandas
    """

    def __init__(self, path: Pathlike):
        self.file_path = pathlib.Path(path)
        self.extension = self.file_path.suffix[1:]

    def _dump_data(self, **kwargs) -> pd.DataFrame:
        return getattr(pd, f"read_{self.extension}")(self.file_path, **kwargs)

    def _load_data(self, data, **kwargs):
        return getattr(pd, f"to_{self.extension}")(data, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__} - FileDataset>"
