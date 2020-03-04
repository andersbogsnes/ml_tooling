import abc
import logging
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connectable
from sqlalchemy.exc import DBAPIError
from typing import Optional, Tuple, Union
from contextlib import contextmanager

from ml_tooling.data.base_data import Dataset
from ml_tooling.utils import DataType, DatasetError

logger = logging.getLogger("ml_tooling")


class SQLDataset(Dataset, metaclass=abc.ABCMeta):
    """
    An Abstract Base Class for use in creating SQL Datasets.
    This class is intended to be subclassed and must provide a
    :meth:`load_training_data` and :meth:`load_prediction_data` method.

    These methods must accept a `conn` argument which is an instance of a SQLAlchemy connection.
    This connection will be passed to the method by the SQLDataset at runtime.

    Attributes
    ----------
    table: sa.Table
        SQLAlchemy table definition to use when loading the dataset. Is the table that will be
        copied when using `.copy_to` and should be the canonical definition of the feature set.
        Do not define a schema - that is set at runtime


    Methods
    -------
    load_prediction_data(idx, conn)
        Used to load prediction data for a given idx - returns features

    load_training_data(conn)
        Used to load the full training dataset - returns features and targets

    """

    table: Optional[sa.Table] = None

    def __init__(self, conn: Union[str, Connectable], schema: str, **kwargs):
        """
        Instantiates a dataset with the necessary arguments to connect to the database.

        Parameters
        ----------
        conn: Connectable
            Either a valid DB_URL string or an engine to connect to the database
        schema: str
            A string naming the schema to use - allows for swapping schemas at runtime
        kwargs: dict
            Kwargs are passed to `create_engine` if conn is a string
        """
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
    def create_connection(self) -> sa.engine.Connection:
        """
        Instantiates a connection to be used in reading and writing to the database.

        Ensures that connections are closed properly and dynamically inserts the schema
        into the database connection

        Returns
        -------
        sa.engine.Connection
            An open connection to the database, with a dynamically defined schema
        """
        conn = self.engine.connect().execution_options(
            schema_translate_map={None: self.schema}
        )
        try:
            yield conn
        finally:
            conn.close()

    @abc.abstractmethod
    def load_training_data(
        self, conn, *args, **kwargs
    ) -> Tuple[pd.DataFrame, DataType]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_prediction_data(self, idx, conn, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
        with self.create_connection() as conn:
            return super()._load_training_data(*args, conn=conn, **kwargs)

    def _load_prediction_data(self, *args, **kwargs) -> pd.DataFrame:
        with self.create_connection() as conn:
            return super()._load_prediction_data(*args, conn=conn, **kwargs)

    def _dump_data(self, use_cache=False) -> pd.DataFrame:
        """
        Reads the underlying SQL table and returns a DataFrame

        Returns
        -------
        pd.DataFrame

        """

        logger.info(f"Dumping data from {self.table}")
        logger.debug(f"Dumping data from {self.engine}/{self.table.name}")
        stmt = sa.select([self.table])
        try:
            with self.create_connection() as conn:
                data = pd.read_sql(stmt, conn)
        except DBAPIError:
            logger.exception("Data dump failed")
            raise

        logger.info("Data dumped...")
        return data

    def _setup_table(self, conn: sa.engine.Connection):
        """
        Sets up a clean table with all necessary schemas created

        Parameters
        ----------
        conn: sa.engine.Connection
            Connection to the database
        """
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(sa.schema.CreateSchema(self.schema))
        self.table.drop(bind=conn, checkfirst=True)
        self.table.create(bind=conn)

    def _load_data(self, data: pd.DataFrame):
        """
        Writes input data to the underlying SQL Table

        Parameters
        ----------
        data: pd.DataFrame
            Input data to write to file
        """
        logger.info(f"Inserting data into {self.table.name}")
        with self.create_connection() as conn:
            trans = conn.begin()
            try:
                self._setup_table(conn)
                data.to_sql(
                    self.table.name,
                    conn,
                    schema=self.schema,
                    if_exists="append",
                    index=False,
                )
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
