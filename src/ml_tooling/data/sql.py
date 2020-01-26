import abc
import logging
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connectable
from sqlalchemy.exc import DBAPIError
from typing import Optional, Tuple
from contextlib import contextmanager

from ml_tooling.data.base_data import Dataset
from ml_tooling.utils import DataType, DatasetError


logger = logging.getLogger("ml_tooling")


class SQLDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Baseclass for creating SQL based Datasets. Subclass SQLDataset and provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method.
    SQLDataset takes a SQLAlchemy connection object or string URI to create the engine
    """

    table: Optional[sa.Table] = None

    def __init__(self, conn: Connectable, schema: str, **kwargs):
        if isinstance(conn, Connectable):
            self.engine = conn
        elif isinstance(conn, str):
            self.engine = sa.create_engine(conn, **kwargs)
        else:
            raise ValueError(f"Invalid connection")
        if self.table is not None and self.table.schema is not None:
            raise DatasetError(
                f"{self.table.schema.name} cannot have a defined schema - "
                f"remove the schema declaration"
            )
        self.schema = schema

    @contextmanager
    def create_connection(self):
        conn = self.engine.connect().execution_options(
            schema_translate_map={None: self.schema}
        )
        try:
            yield conn
        finally:
            conn.close()

    def _load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
        with self.create_connection() as conn:
            return self.load_training_data(conn)

    def _load_prediction_data(self, *args, **kwargs) -> pd.DataFrame:
        with self.create_connection() as conn:
            return self.load_prediction_data(conn)

    def _dump_data(self):
        logger.info(f"Dumping data from {self.table}")
        logger.debug(f"Dumping data from {self.engine}/{self.table.name}")
        stmt = sa.select([self.table])
        try:
            with self.create_connection() as conn:
                data = conn.execute(stmt).fetchall()
        except DBAPIError:
            logger.exception("Data dump failed")
            raise

        logger.info("Data dumped...")
        return data

    def _setup_table(self, conn: sa.engine.Connection):
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(sa.schema.CreateSchema(self.schema))
        self.table.drop(bind=conn, checkfirst=True)
        self.table.create(bind=conn)

    def _load_data(self, data):
        logger.info(f"Inserting data into {self.table.name}")
        with self.create_connection() as conn:
            trans = conn.begin()
            try:
                self._setup_table(conn)
                insert = self.table.insert()
                conn.execute(insert, data)
                trans.commit()
            except DBAPIError:
                logger.exception("Insert data failed")
                trans.rollback()
                raise

    def copy_to(self, target: "SQLDataset") -> "SQLDataset":
        """
        Copies data from one database table into other. This will truncate the table and
        load new data in.

        Parameters
        ----------
        target: SQLDataset
            A SQLDataset object representing the table you want to copy the data into

        Returns
        -------
        SQLDataset
            The target dataset to copy to
        """
        logger.info("Copying data...")
        logger.debug(
            f"Copying data from {self.engine}/{self.table.name} into " f"{target}"
        )
        data = self._dump_data()
        target._load_data(data)
        return target

    def __repr__(self):
        return f"<{self.__class__.__name__} - SQLDataset {self.engine}>"
