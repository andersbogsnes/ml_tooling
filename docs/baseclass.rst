.. py:currentmodule:: ml_tooling.baseclass
.. _baseclass:
.. _model:

Model
=====

The Model baseclass contains all the neat functionality of ML Tooling.

In order to take advantage of this functionality, simply wrap your Scikit-learn model
using the Model class.

We will be using scikit-learn's built-in Boston houseprices dataset and try to fit a
:class:`~sklearn.linear_model.LinearRegression`

.. seealso::
    :ref:`api` for a full overview of methods

Example Usage
-------------
First we need to define how we want to load our data. This is done by defining a
:class:`~ml_tooling.data.Dataset` class and creating the
:meth:`~ml_tooling.data.Dataset.load_training_data`
and :meth:`~ml_tooling.data.Dataset.load_prediction_data` methods. In this example, we use
the Boston dataset from `sklearn.datasets`

We then simply wrap a :class:`~sklearn.linear_model.LinearRegression` using our
:class:`Model` class and we are ready to begin!

.. doctest::

    >>> from ml_tooling import Model
    >>> from ml_tooling.data import Dataset
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> # Define our DataClass
    >>> class BostonData(Dataset):
    ...     # Tell the Dataset how to load data when training
    ...     def load_training_data(self):
    ...         return load_boston(return_X_y=True)
    ...
    ...     # Tell the Dataset how to load data when predicting
    ...     # In this example, we want to predict a single house at a time
    ...     def load_prediction_data(self, idx):
    ...         x, _ = load_boston(return_X_y=True)
    ...         return x[idx]
    >>>
    >>> # Now we can create our dataset and our model
    >>> linear = Model(LinearRegression())
    >>> bostondata = BostonData()
    >>> # Remember to setup a train test split!
    >>> bostondata.create_train_test()
    <Dataset BostonData>


Configuration
-------------

To change the default configuration values, modify the :attr:`~Model.config` attributes directly:

.. doctest::

    >>> linear.config.RANDOM_STATE = 2

.. seealso::
    :ref:`config` for a list of available configuration options

Logging
-------

We also have the ability to log our experiments using the :meth:`Model.log` context manager.

.. doctest::

    >>> with linear.log('test_dir'):
    ...     linear.score_estimator(bostondata)
    <Result LinearRegression: r2: 0.68>

.. testcleanup::

    import shutil
    shutil.rmtree(linear.config.RUN_DIR.joinpath('test_dir'))

This will write a yaml file specifying attributes of the model, results, git-hash of the model
and other pertinent information.

.. seealso::

    Check out :meth:`Model.log` for more info on what is logged




Continue to :doc:`plotting`
