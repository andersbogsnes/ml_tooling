import abc
from typing import Tuple

import sqlalchemy as sa
from sqlalchemy.engine import Connectable

from ml_tooling.data.base_data import DataSet
from ml_tooling.utils import DataType


class SQLDataSet(DataSet, metaclass=abc.ABCMeta):
    def __init__(self, conn, **engine_kwargs):
        if isinstance(conn, Connectable):
            self.engine = conn
        elif isinstance(conn, str):
            self.engine = sa.create_engine(conn, **engine_kwargs)
        else:
            raise ValueError(f"Invalid connection")

    @abc.abstractmethod
    def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
        pass

    @abc.abstractmethod
    def load_prediction_data(self, *args, **kwargs) -> DataType:
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} - SQLDataset>"
