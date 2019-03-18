API
===

.. _api:

BaseClassModel
--------------
Abstract BaseClass to inherit from.
Implements all the base functionality needed to create the wrapper

.. autoclass:: ml_tooling.baseclass.BaseClassModel
    :members:

    .. automethod:: log(self, run_name)

    .. automethod:: setup_model()

        To be implemented by the user - setup_model is a classmethod which loads up an untrained model.
        Typically this would setup a pipeline and the selected model for easy training

        Returning to our previous example of the BostonModel, let us implement a setup_model method

    .. code-block:: python

        from ml_tooling import BaseClassModel
        from sklearn.datasets import load_boston
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        import pandas as pd

        class BostonModel(BaseClassModel):
            def get_prediction_data(self, idx):
                data = load_boston()
                df = pd.DataFrame(data=data.data, columns=data.feature_names)
                return df.iloc[idx] # Return given observation

            def get_training_data(self):
                data = load_boston()
                return pd.DataFrame(data=data.data, columns=data.feature_names), data.target

            @classmethod
            def setup_model(cls):
                pipeline = Pipeline([('scaler', StandardScaler()),
                                     ('clf', LinearRegression())
                                     ])
                return cls(pipeline)

    Given this extra setup, it becomes easy to load the untrained model to train it::

        model = BostonModel.setup_model()
        model.train_model()

.. include:: config.inc.rst
