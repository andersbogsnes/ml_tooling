.. ML Tooling documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:39:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ML Tooling!
======================================
ML Tooling is a framework for working with scikit-learn compatible models. We support Python 3.6+

Why?
====
It standardises interfaces for training, saving and loading models while also providing a number of
utility functions for everyday tasks such as plotting, comparing results and gridsearching.
ML Tooling was developed at `Alm Brand <http://www.almbrand.dk>`_

All you have to do to get started is to define your input data:

.. code-block:: python

   from ml_tooling import BaseClassModel
   from sklearn.ensemble import RandomForestClassifier

   # Define your data input
   class DataModel(BaseClassModel):
      def get_training_data(self):
         # Load your training data from your source of choice
         return X, y
      def get_prediction_data(self, idx):
         # Load required data for prediction of a given idx
         return X

   # Use your data with a given model
   random_forest = DataModel(RandomForestClassifier())

You now have access to all the utility functions available in ML Tooling

Check out :ref:`example` to see a fully worked example

ML Tooling has three parts:

:doc:`transformers`
###################

We have implemented a number of pandas-compatible Scikit-learn transformers
for use in pipelines

:doc:`baseclass`
################

By defining your DataModel, you get access to a number of utility functions.

:doc:`plotting`
###############

Datascientists always need plots - we've implemented the most common plots here.


Table of Contents
#################

.. toctree::
   :maxdepth: 1

   install.rst
   example.rst
   baseclass.rst
   plotting.rst
   transformers.rst
   api.rst
======================
Welcome to ML Toolings documentation. Get started by :ref:`install` and head over to :ref:`quickstart`
There are three main parts to ML Tooling:

* :ref:`baseclass` wraps all the functionality,

* :ref:`transformer` has a number of pandas-compatible Transformers for use in scikit-learn `Pipelines`_

* :ref:`plotting` has a number of utility plotting functions that can be used standalone


ML Tooling is a convenient wrapper around `Scikit-learn`_ and `Pandas`_
- check the documentation there.


Why?
====
ML Tooling standardises interfaces for training, saving and loading models while also providing a
number of utility functions for everyday tasks such as plotting, comparing results and gridsearching.

ML Tooling was developed at `Alm Brand <http://www.almbrand.dk>`_


.. _Scikit-learn: https://scikit-learn.org/stable/
.. _Pandas: https://pandas.pydata.org/
.. _Pipelines: https://scikit-learn.org/stable/modules/compose.html

.. include:: contents.rst