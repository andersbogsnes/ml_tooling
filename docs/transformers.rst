.. _transformer:

Transformers
============

One great feature of `scikit-learn`_ is the concept of the :class:`~sklearn.pipeline.Pipeline`
alongside :ref:`transformers <sklearn:preprocessing>`

.. _scikit-learn: https://scikit-learn.org
.. _pandas: https://pandas.pydata.org/

By default, scikit-learn's transformers will convert a `pandas`_ :class:`~pandas.DataFrame` to numpy arrays -
losing valuable column information in the process. We have implemented a number of transformers
that accept a `pandas`_ :class:`~pandas.DataFrame` and return a `pandas`_ :class:`~pandas.DataFrame`.

Select
------
A column selector - Provide a list of columns to be passed on in the pipeline

Example
#######
Pass a list of column names to be selected

.. doctest::

    >>> from ml_tooling.transformers import Select
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    "id": [1, 2, 3, 4],
    ...    "status": ["OK", "Error", "OK", "Error"],
    ...    "sales": [2000, 3000, 4000, 5000]
    ... })
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
You can pass any value to replace NaNs with

.. doctest::

    >>> from ml_tooling.transformers import FillNA
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...        "id": [1, 2, 3, 4],
    ...        "sales": [2000, 3000, 4000, np.nan]
    ... })
    >>> fill_na = FillNA(value = 0)
    >>> fill_na.fit_transform(df)
       id   sales
    0   1  2000.0
    1   2  3000.0
    2   3  4000.0
    3   4     0.0


You can also use one of the built-in strategies.

- :code:`mean`
- :code:`median`
- :code:`most_freq`
- :code:`max`
- :code:`min`

.. doctest::

    >>> fill_na = FillNA(strategy='mean')
    >>> fill_na.fit_transform(df)
       id   sales
    0   1  2000.0
    1   2  3000.0
    2   3  4000.0
    3   4  3000.0

In addition, FillNa will indicate if a value in a column was missing if you set `indicate_nan=True`.
This creates a new column of 1 and 0 indicating missing values

.. doctest::

    >>> fill_na = FillNA(strategy='mean', indicate_nan=True)
    >>> fill_na.fit_transform(df)
       id   sales  sales_is_nan
    0   1  2000.0             0
    1   2  3000.0             0
    2   3  4000.0             0
    3   4  3000.0             1

ToCategorical
-------------

Performs one-hot encoding of categorical values through :class:`pandas.Categorical`.
All categorical values not found in training data will be set to 0

Example
#######

.. doctest::

    >>> from ml_tooling.transformers import ToCategorical
    >>> df = pd.DataFrame({
    ...    "status": ["OK", "Error", "OK", "Error"]
    ... })
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
- under the hood, FuncTransformer uses :meth:`~pandas.DataFrame.apply`

.. doctest::

    >>> from ml_tooling.transformers import FuncTransformer
    >>> df = pd.DataFrame({
    ...    "status": ["OK", "Error", "OK", "Error"]
    ... })
    >>> uppercase = FuncTransformer(lambda x: x.str.upper())
    >>> uppercase.fit_transform(df)
      status
    0     OK
    1  ERROR
    2     OK
    3  ERROR

FuncTransformer also supports passing keyword arguments to the function

.. doctest::

    >>> from ml_tooling.transformers import FuncTransformer
    >>> def custom_func(input, word1, word2):
    ...    result = ""
    ...    if input == "OK":
    ...       result = word1
    ...    elif input == "Error":
    ...       result = word2
    ...    return result
    >>> def wrapper(df, word1, word2):
    ...   return df.apply(custom_func,args=(word1,word2))
    >>> df = pd.DataFrame({
    ...     "status": ["OK", "Error", "OK", "Error"]
    ... })
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

Here we want to bin our sales data into 3 buckets
.. doctest::

    >>> from ml_tooling.transformers import Binner
    >>> df = pd.DataFrame({
    ...    "sales": [1500, 2000, 2250, 7830]
    ... })
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

.. doctest::

    >>> from ml_tooling.transformers import Renamer
    >>> df = pd.DataFrame({
    ...     "Total Sales": [1500, 2000, 2250, 7830]
    ... })
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

.. doctest::

    >>> from ml_tooling.transformers import DateEncoder
    >>> df = pd.DataFrame({
    ...     "sales_date": [pd.to_datetime('2018-01-01'), pd.to_datetime('2018-02-02')]
    ... })
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
.. doctest::

    >>> from ml_tooling.transformers import FreqFeature
    >>> df = pd.DataFrame({
    ...     "sales_category": ['Sale', 'Sale', 'Not Sale']
    ... })
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

.. doctest::

    >>> from ml_tooling.transformers import FreqFeature, Binner, Select, DFFeatureUnion
    >>> from sklearn.pipeline import Pipeline
    >>> df = pd.DataFrame({
    ...     "sales_category": ['Sale', 'Sale', 'Not Sale', 'Not Sale'],
    ...     "sales": [1500, 2000, 2250, 7830]
    ... })
    >>> freq = Pipeline([
    ...     ('select', Select('sales_category')),
    ...     ('freq', FreqFeature())
    ... ])
    >>> binned = Pipeline([
    ...     ('select', Select('sales')),
    ...     ('bin', Binner(bins=[0, 1000, 2000, 8000]))
    ...     ])
    >>> union = DFFeatureUnion([
    ...    ('sales_category', freq),
    ...    ('sales', binned)
    ... ])
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

.. doctest::

    >>> from ml_tooling.transformers import DFRowFunc
    >>> df = pd.DataFrame({
    ...    "number_1": [1, np.nan, 3, 4],
    ...    "number_2": [1, 3, 2, 4]
    ... })
    >>> rowfunc = DFRowFunc(strategy = 'sum')
    >>> rowfunc.fit_transform(df)
         0
    0  2.0
    1  3.0
    2  5.0
    3  8.0


You can also use any callable that takes a :class:`pandas.Series`

.. doctest::

    >>> rowfunc = DFRowFunc(strategy = np.mean)
    >>> rowfunc.fit_transform(df)
         0
    0  1.0
    1  3.0
    2  2.5
    3  4.0


Binarize
--------
Convenience transformer which returns 1 where the column value is equal to given value else 0.

Example
#######

.. doctest::

    >>> from ml_tooling.transformers import Binarize
    >>> df = pd.DataFrame({
    ...     "number_1": [1, np.nan, 3, 4],
    ...     "number_2": [1, 3, 2, 4]
    ... })
    >>> binarize = Binarize(value = 3)
    >>> binarize.fit_transform(df)
       number_1  number_2
    0         0         0
    1         0         1
    2         1         0
    3         0         0


RareFeatureEncoder
------------------
Replaces categories with a value, if they occur less than a threshold. - Using :meth:`pandas.Series.value_counts()`.
The fill value can be any value and the threshold can be either a percent or int value.

The column names needs to be identical when using Train & Test dataset

The Transformer does not count NaN.

Example
#######

.. doctest::

    >>> from ml_tooling.transformers import RareFeatureEncoder
    >>> df = pd.DataFrame({
    ...         "categorical_a": [1, "a", "a", 2, "b", np.nan],
    ...         "categorical_b": [1, 2, 2, 3, 3, 3],
    ...         "categorical_c": [1, "a", "a", 2, "b", "b"],
    ... })

    >>> rare = RareFeatureEncoder(threshold=2, fill_rare="Rare")
    >>> rare.fit_transform(df)
        categorical_a categorical_b categorical_c
    0	         Rare	       Rare          Rare
    1	            a	          2             a
    2	            a	          2             a
    3	         Rare             3          Rare
    4	         Rare             3             b
    5	          NaN	          3             b
