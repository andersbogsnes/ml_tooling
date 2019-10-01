import abc

from ml_tooling.data.base_data import Dataset
import sqlalchemy as sa
from sqlalchemy.engine import Connectable


class SQLDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Baseclass for creating SQL based Datasets. Subclass SQLDataset and provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method.
    SQLDataset takes a SQLAlchemy connection object or string URI to create the engine
    """

    def __init__(self, conn, **engine_kwargs):
        if isinstance(conn, Connectable):
            self.engine = conn
        elif isinstance(conn, str):
            self.engine = sa.create_engine(conn, **engine_kwargs)
        else:
            raise ValueError(f"Invalid connection")

    def __repr__(self):
        return f"<{self.__class__.__name__} - SQLDataset>"
