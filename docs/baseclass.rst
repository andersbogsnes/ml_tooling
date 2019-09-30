.. py:currentmodule:: ml_tooling.baseclass
.. _baseclass:

Model
=========

The Model contains all the neat functionality of ML Tooling.

In order to take advantage of this functionality, simply wrap your Scikit-learn model
using the Model class. We will be using scikit-learn's built-in Boston houseprices
dataset for demo purposes and we just want to try it out with a simple linear regression

.. seealso::
    :ref:`api` for a full overview of methods

Example Usage
-------------
First we need to define how we want to load our data. This is done by defining a
:class:`~ml_tooling.data.Dataset` class and creating the
:meth:`~ml_tooling.data.Dataset.load_training_data`
and :meth:`~ml_tooling.data.Dataset.load_prediction_data` methods. In this example, we use
the Boston dataset from `sklearn.datasets`

We then simply wrap a LinearRegression using our Model class and we are ready to begin!

.. doctest::

    from ml_tooling import Model
    from ml_tooling.data import Dataset
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    # Define our DataClass
    class BostonData(Dataset):
        # Tell the Dataset how to load data when training
        def get_training_data(self):
            return load_boston(return_X_y=True)

        # Tell the Dataset how to load data when predicting
        # In this example, we want to predict a single house at a time
        def get_prediction_data(self, idx):
            x, _ = load_boston(return_X_y=True)
            return x[idx]

    # Now we can create our dataset and our model
    linear = Model(LinearRegression())
    bostondata = BostonData()


Configuration
-------------

To change the default configuration values, modify the :attr:`~ModelData.config` attributes directly::

    linear.config.RANDOM_STATE = 2

.. seealso::
    :ref:`config` for a list of available configuration options

Continue to :doc:`plotting`
