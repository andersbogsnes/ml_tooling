.. _tutorial:


Tutorial
========
Let's go through a more involved example from start to end reusing our Boston Houseprice dataset from before

We start with our normal imports

>>> from ml_tooling import Model
>>> from ml_tooling.data import Dataset
>>> from sklearn.datasets import load_boston
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.linear_model import LinearRegression, Ridge

Continue to :doc:`baseclass`
