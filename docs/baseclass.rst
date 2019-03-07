.. _baseClass:

BaseClass
=========

The BaseClass contains all the neat functionality of ML Tooling.

In order to take advantage of this functionality, start by creating your DataClass
which inherits from BaseClass. We will be using scikit-learn's built-in Boston houseprices
dataset for demo purposes and we just want to try it out with a simple linear regression

Your subclass must implement two methods:
- ``get_prediction_data()``



.. code-block:: python

    from ml_tooling import BaseClassModel
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression

    # Define our DataClass
    class BostonData(BaseClassModel):
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
