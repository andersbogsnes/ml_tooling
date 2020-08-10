Dataset
=======

SQLDataset
----------

Creating a SQLDataset from a table in a db
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~ml_tooling.data.SQLDataset` should be used for creating datasets based on SQL database source.
SQLDatasets must be provided with a :class:`sqlalchemy.engine.Connectable` or a valid connection string.

When writing the :meth:`~ml_tooling.data.SQLDataset.load_training_data`
and :meth:`~ml_tooling.data.SQLDataset.load_prediction_data`, they must accept a connection in their arguments - this
will be provided at runtime by the SQLDataset.

.. code-block::

    >>> import pandas as pd
    >>> from ml_tooling.data import SQLDataset
    >>> from sqlalchemy import create_engine
    >>>
    >>> class TableData(SQLDataset):
    ...
    ...     table = "TableName"
    ...
    ...     def load_training_data(self, conn):
    ...         return pd.read_sql(table, conn)
    ...
    ...     def load_prediction_data(self, conn, idx):
    ...         return pd.read_sql(table, conn).loc[idx]
    >>>
    >>> engine = create_engine("postgresql://username:password@localhost/test")
    >>>
    >>> TableData(engine)
    <TableData - SQLDataset>


FileDataset
-----------

Creating a FileDataset from a csv file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When we create a FileDataset, we need to specify the location of our datafiles -
this will be available in the :code:`self.file_path` attribute. ML Tooling can
A more elaborate example of using this dataset can be found at :file:`../notebooks/Titanic Demo.ipynb`.

.. code-block::

    >>> import pandas as pd
    >>> from ml_tooling.data import FileDataset
    >>>
    >>> class TitanicData(FileDataset):
    ...     def load_training_data(self):
    ...         data = self.read_file()
    ...         return data.drop(columns='Survived'), data.Survived
    ...
    ...     def load_prediction_data(self, idx):
    ...         data = self.read_file()
    ...         return data.drop(columns="Survived").idx[[idx]]
    >>>
    >>> TitanicData("./titanic.csv")
    <TitanicData - FileDataset>

When a Dataset is correctly defined, you can use all the methods defined in :class:`~ml_tooling.data.Dataset`

Copying Datasets
----------------

If you have two datasets defined, you can copy data from one into the other. For example, if you have defined
a SQLDataset and want to copy it into a file:

.. code-block::

    >>> source_data = TitanicSQLData("postgresql://localhost:5432", schema="prod")
    >>> target_data = TitanicFileData("./titanic.csv")
    >>> source_data.copy_to(target_data)

This will read the data from the SQL database and write it to a csv file named ``titanic.csv``

A common usecase for this is to move data from a central datastore into a local datastore, keeping two
database tables in sync.

.. code-block::

    >>> source_data = TitanicSQLData("postgresql://my-prod-database", schema="prod")
    >>> target_data = TitanicSQLData("postgresql://my-api-database", schema="public")
    >>> source_data.copy_to(target_data)

Demo Datasets
----------------

If you want to test your model on a demo datasets from :ref:`sklearn:datasets`, you can use the function
:func:`~ml_tooling.data.load_demo_dataset`

.. doctest::

    >>> from ml_tooling.data import load_demo_dataset
    >>>
    >>> bostondata = load_demo_dataset("boston")
    >>> # Remember to setup a train test split!
    >>> bostondata.create_train_test()
    <BostonData - Dataset>
