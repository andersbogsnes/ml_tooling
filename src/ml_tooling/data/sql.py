from typing import Union

import pandas as pd

try:
    import sqlalchemy as sa
    from sqlalchemy.sql import Selectable
except ImportError:
    raise ImportError("SQLAlchemy is required. Please install SQLAlchemy.")


class SQLBase:
    def __init__(self, conn_string):
        self.engine = sa.create_engine(conn_string)


class SQLDataSet:
    def __init__(self, conn_string, table: sa.Table, **engine_kwargs):
        super().__init__()
        self.engine = sa.create_engine(conn_string, **engine_kwargs)
        self.table = table

    def read(self, filter: Selectable = None):
        return pd.read_sql(self.table.select(whereclause=filter), self.engine)

    def save(self, data: pd.DataFrame):
        self.table.drop(self.engine)
        self.table.create(self.engine, checkfirst=True)

        insert_data = data.to_dict(orient="records")
        self.engine.execute(self.table.insert(), insert_data)


class QueryDataSet:
    def __init__(self, conn_string, query: Union[Selectable, str], **engine_kwargs):
        self.engine = sa.create_engine(conn_string, **engine_kwargs)
        self.query = query

    def read(self):
        return
