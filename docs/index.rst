.. ML Tooling documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:39:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ML Tooling!
======================

.. figure:: images/ML_logo.png
   :width: 600
   :alt: Logo for ML Tooling
   :align: left

Welcome to ML Tooling's documentation!

Get started by :ref:`installing <install>` and head over to :ref:`quickstart`!

There are three main parts to ML Tooling:

* :ref:`baseclass` wraps all the functionality,

* :ref:`transformer` has a number of pandas-compatible scikit-learn `Transformers`_ for use in scikit-learn `Pipelines`_

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
.. _Transformers: https://scikit-learn.org/stable/modules/preprocessing.html

.. include:: contents.rst