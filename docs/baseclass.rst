.. py:currentmodule:: ml_tooling.baseclass
.. _baseclass:

BaseClass
=========

The BaseClass contains all the neat functionality of ML Tooling.

In order to take advantage of this functionality, start by creating your DataClass
which inherits from BaseClass. We will be using scikit-learn's built-in Boston houseprices
dataset for demo purposes and we just want to try it out with a simple linear regression

Your subclass must implement two methods:

- :meth:`~ModelData.get_training_data`

Method that retrieves all training data. Used for training and evaluating the model


- :meth:`~ModelData.get_prediction_data`

Method that, given an input, fetches corresponding features. Used for predicting an unseen observation

.. seealso::
    :ref:`api` for a full overview of methods

Example Usage
-------------
We define a class using ModelData and implement the two required methods.
In this example, we implement a linear regression on the Boston dataset using sklearn.datasets

.. code-block:: python

    from ml_tooling import ModelData
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    # Define our DataClass
    class BostonData(ModelData):
        # Tell the DataModel how to load data when training
        def get_training_data(self):
            return load_boston(return_X_y=True)

        # Tell the DataModel how to load data when predicting
        # In this example, we want to predict a single house at a time
        def get_prediction_data(self, idx):
            x, _ = load_boston(return_X_y=True)
            return x[idx]

    # Now we can instantiate our DataModel with our estimator
    linear_boston = BostonData(LinearRegression())

You can also instantiate a `BostonData` model without an estimator, but before you can use it
to do any estimation, you need to use the :meth:`~ModelData.init_estimator` to set an estimator.

.. code-block:: python

    boston = BostonData()
    boston.init_estimator(LinearRegression())


Configuration
-------------

To change the default configuration values, modify the :attr:`~ModelData.config` attributes directly::

    BostonData.config.RANDOM_STATE = 2

That will update the configuration for all instances of BostonData

.. seealso::
    :ref:`config` for a list of available configuration options

Continue to :doc:`plotting`
