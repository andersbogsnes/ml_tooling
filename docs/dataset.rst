Dataset
=======

SQLDataset
----------

Creating a SQLDataset from a table in a db
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~ml_tooling.data.SQLDataset` should be used for making datasets that get their data from a SQL based source.
SQLDatasets must be provided with a :py:class:`sqlalchemy.engine.Connectable` or a valid connection string.


.. code-block::

    >>> import pandas as pd
    >>> from ml_tooling.data import SQLDataset
    >>> from sqlalchemy import create_engine
    >>>
    >>> class TableData(SQLDataset):
    ...
    ...     table = "TableName"
    ...
    ...     def load_training_data(self):
    ...         return pd.read_sql_table(table, self.engine)
    ...
    ...     def load_prediction_data(self, idx):
    ...         return pd.read_sql_table(table, self.engine).loc(idx)
    >>>
    >>> engine = create_engine("postgresql://username:password@localhost/test")
    >>>
    >>> TableData(engine)
    <TableData - SQLDataset>


FileDataset
-----------

Creating a FileDataset from a csv file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a FileDataset, one should define how the data is loaded based on the :code:`self.file_path` variable
A more elaborate example of using this dataset can be found at :file:`../notebooks/Titanic Demo.ipynb`.

.. code-block::

    >>> import pandas as pd
    >>> from ml_tooling.data import FileDataset
    >>>
    >>> class TitanicData(FileDataset):
    ...     def load_training_data(self):
    ...         data = pd.read_csv(self.file_path / "train.csv")
    ...         return data.drop('Survived', axis=1), data.Survived
    ...
    ...     def load_prediction_data(self):
    ...         data = pd.read_csv(self.file_path / "test.csv")
    ...         return data
    >>>
    >>> TitanicData("./data/raw")
    <TitanicData - FileDataset>

When a Dataset is correctly defined, one can use supplementing methods defined in :class:`~ml_tooling.data.Dataset`
