.. _transformer:

Transformers
============

One great feature of scikit-learn is the concept of the :class:`sklearn.pipeline.Pipeline`
alongside `transformers`_

.. _transformers: https://scikit-learn.org/stable/modules/preprocessing.html

By default, scikit-learn's transformers will convert a pandas :class:`~pandas.DataFrame` to numpy arrays -
losing valuable column information in the process. We have implemented a number of transformers
that accept a pandas :class:`~pandas.DataFrame` and return a pandas :class:`~pandas.DataFrame`.

Select
------
A column selector - Provide a list of columns to be passed on in the pipeline

Example
#######
Pass a list of column names to be selected::

    from ml_tooling.transformers import Select
    import pandas as pd

    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "status": ["OK", "Error", "OK", "Error"],
        "sales": [2000, 3000, 4000, 5000]

    })

    >>> select = Select(['id', 'status'])
    >>> select.fit_transform(df)
       id status
    0   1     OK
    1   2  Error
    2   3     OK
    3   4  Error



FillNA
------

Fills NA values with given value or strategy. Either a value or a strategy has to be supplied.

Examples
########
You can pass any value to replace NaNs with::

    from ml_tooling.transformers import FillNA
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "status": ["OK", "Error", "OK", "Error"],
        "sales": [2000, 3000, 4000, np.nan]

    })

    >>> fill_na = FillNA(value = 0)
    >>> fill_na.fit_transform(df)
       id status   sales
    0   1     OK  2000.0
    1   2  Error  3000.0
    2   3     OK  4000.0
    3   4  Error     0.0


You can also use one of the built-in strategies.

- 'mean'
- 'median'
- 'most_freq'
- 'max'
- 'min'

>>> fill_na = FillNA(value='mean')
>>> fill_na.fit_transform(df)
   id status   sales
0   1     OK  2000.0
1   2  Error  3000.0
2   3     OK  4000.0
3   4  Error  3000.0

ToCategorical
-------------

Performs one-hot encoding of categorical values through :class:`~pandas.Categorical`.
All categorical values not found in training data will be set to 0

Example
#######

.. code-block:: python

    from ml_tooling.transformers import ToCategorical
    import pandas as pd

    df = pd.DataFrame({
        "status": ["OK", "Error", "OK", "Error"]

    })

    >>> onehot = ToCategorical()
    >>> onehot.fit_transform(df)
       status_Error  status_OK
    0             0          1
    1             1          0
    2             0          1
    3             1          0


FuncTransformer
---------------
Applies a given function to each column

Example
#######
We can use any arbitrary function that accepts a :class:`pandas.Series`
- under the hood, FuncTransformer uses :meth:`~pandas.DataFrame.apply`::

    from ml_tooling.transformers import FuncTransformer
    import pandas as pd

    df = pd.DataFrame({
        "status": ["OK", "Error", "OK", "Error"]
    })

>>> uppercase = FuncTransformer(lambda x: x.str.upper())
>>> uppercase.fit_transform(df)
  status
0     OK
1  ERROR
2     OK
3  ERROR

FuncTransformer also supports passing keyword arguments to the function::


    from ml_tooling.transformers import FuncTransformer
    import pandas as pd

    def custom_func(input, word1, word2):
       result = ""
       if input == "OK":
          result = word1
       elif input == "Error":
          result = word2
       return result

    def wrapper(df, word1, word2):
       return df.apply(custom_func,args=(word1,word2))

    df = pd.DataFrame({
        "status": ["OK", "Error", "OK", "Error"]
    })

    >>> kwargs = {'word1': 'Okay','word2': 'Fail'}
    >>> wordchange = FuncTransformer(wrapper,**kwargs)
    >>> wordchange.fit_transform(df)
      status
    0   Okay
    1   Fail
    2   Okay
    3   Fail

Binner
------
Bins numerical data into supplied bins. Bins are passed on to :func:`pandas.cut`

Example
-------

Here we want to bin our sales data into 3 buckets::

    from ml_tooling.transformers import Binner
    import pandas as pd

    df = pd.DataFrame({
        "sales": [1500, 2000, 2250, 7830]
    })

    >>> binned = Binner(bins=[0, 1000, 2000, 8000])
    >>> binned.fit_transform(df)
              sales
    0  (1000, 2000]
    1  (1000, 2000]
    2  (2000, 8000]
    3  (2000, 8000]

Renamer
-------
Renames columns to be equal to the passed list - must be in order

Example
########

.. code-block:: python

    from ml_tooling.transformers import Renamer
    import pandas as pd

    df = pd.DataFrame({
        "Total Sales": [1500, 2000, 2250, 7830]
    })


    >>> rename = Renamer(['sales'])
    >>> rename.fit_transform(df)
       sales
    0   1500
    1   2000
    2   2250
    3   7830


DateEncoder
-----------
Adds year, month, day and week columns based on a datefield.
Each date type can be toggled in the initializer

Example
#######

.. code-block:: python

    from ml_tooling.transformers import DateEncoder
    import pandas as pd

    df = pd.DataFrame({
        "sales_date": [pd.to_datetime('2018-01-01'), pd.to_datetime('2018-02-02')]
    })

    >>> dates = DateEncoder(week=False)
    >>> dates.fit_transform(df)
       sales_date_day  sales_date_month  sales_date_year
    0               1                 1             2018
    1               2                 2             2018

FreqFeature
-----------
Converts a column into a normalized frequency

Example
#######
.. code-block:: python

    from ml_tooling.transformers import FreqFeature
    import pandas as pd

    df = pd.DataFrame({
        "sales_category": ['Sale', 'Sale', 'Not Sale']
    })

    >>> freq = FreqFeature()
    >>> freq.fit_transform(df)
       sales_category
    0        0.666667
    1        0.666667
    2        0.333333


DFFeatureUnion
--------------
A FeatureUnion equivalent for DataFrames. Concatenates the result of multiple transformers

Example
#######

.. code-block:: python

    from ml_tooling.transformers import FreqFeature, Binner, Select, DFFeatureUnion
    from sklearn.pipeline import Pipeline
    import pandas as pd


    df = pd.DataFrame({
        "sales_category": ['Sale', 'Sale', 'Not Sale', 'Not Sale'],
        "sales": [1500, 2000, 2250, 7830]
    })


    freq = Pipeline([
        ('select', Select('sales_category')),
        ('freq', FreqFeature())
    ])

    binned = Pipeline([
        ('select', Select('sales')),
        ('bin', Binner(bins=[0, 1000, 2000, 8000]))
        ])


    >>> union = DFFeatureUnion([
    >>>    ('sales_category', freq),
    >>>    ('sales', binned)
    >>> ])
    >>> union.fit_transform(df)
       sales_category         sales
    0             0.5  (1000, 2000]
    1             0.5  (1000, 2000]
    2             0.5  (2000, 8000]
    3             0.5  (2000, 8000]


DFRowFunc
---------
Row-wise operation on :class:`pandas.DataFrame`. Strategy can either be one of the predefined or a callable.
If some elements in the row are NaN these elements are ignored for the built-in strategies.
The built-in strategies are 'sum', 'min' and 'max'

Example
#######

.. code-block:: python

    from ml_tooling.transformers import DFRowFunc
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "number_1": [1, np.nan, 3, 4],
        "number_2": [1, 3, 2, 4]

    })

    >>> rowfunc = DFRowFunc(strategy = 'sum')
    >>> rowfunc.fit_transform(df)
             0
    0        2
    1        3
    2        5
    3        8


You can also use any callable that takes a :class:`pandas.Series`

>>> rowfunc = DFRowFunc(strategy = np.mean)
>>> rowfunc.fit_transform(df)
         0
0        1
1        3
2        2.5
3        4


Binarize
--------
Convenience transformer which returns 1 where the column value is equal to given value else 0.

Example
#######

.. code-block:: python

    from ml_tooling.transformers import Binarize
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "number_1": [1, np.nan, 3, 4],
        "number_2": [1, 3, 2, 4]

    })

    >>> binarize = Binarize(value = 3)
    >>> binarize.fit_transform(df)
             number_1    number_2
    0               0           0
    1               1           0
    2               0           1
    3               0           0
