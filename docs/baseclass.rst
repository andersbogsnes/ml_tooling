.. py:currentmodule:: ml_tooling.baseclass
.. _baseclass:
.. _model:

Model
=====

The :class:`Model` baseclass contains all the neat functionality of ML Tooling.

In order to take advantage of this functionality, simply wrap a model that follows the `scikit-learn`_ API
using the Model class.

We will be using `scikit-learn's <scikit-learn>`_ built-in :func:`Boston <sklearn.datasets.load_boston>`
houseprices dataset to demonstrate how to use ML Tooling.

.. seealso::
    Refer to :ref:`api` for a full overview of methods

First we need to define how we want to load our data. This is done by defining a
:class:`~ml_tooling.data.Dataset` class and creating the
:meth:`~ml_tooling.data.Dataset.load_training_data`
and :meth:`~ml_tooling.data.Dataset.load_prediction_data` methods.

We then simply wrap a :class:`~sklearn.linear_model.LinearRegression` using our
:class:`Model` class and we are ready to begin!

.. doctest::

    >>> from ml_tooling.data import Dataset
    >>> from sklearn.datasets import load_boston
    >>> import pandas as pd
    >>>
    >>> class BostonData(Dataset):
    ...    def load_training_data(self):
    ...        data = load_boston()
    ...        return pd.DataFrame(data.data, columns=data.feature_names), data.target
    ...
    ...    # Define where to get prediction time data - returning a DataFrame
    ...    def load_prediction_data(self, idx):
    ...        data = load_boston()
    ...        x = pd.DataFrame(data.data, columns=data.feature_names)
    ...        return x.loc[[idx]] # Return given observation
    >>>
    >>> bostondata = BostonData()
    >>> # Remember to setup a train test split!
    >>> bostondata.create_train_test()
    <BostonData - Dataset>

Creating your model
~~~~~~~~~~~~~~~~~~~

The first thing to do after creating a dataset object is to create a model object.
This is done by supplying an estimator to the :class:`~ml_tooling.baseclass.Model`.

.. doctest::

    >>> from ml_tooling import Model
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> linear = Model(LinearRegression())
    >>> linear
    <Model: LinearRegression>

Scoring your model
~~~~~~~~~~~~~~~~~~

In order to evaluate the performance of the model use the :meth:`~ml_tooling.baseclass.Model.score_estimator` method.
This will train the estimator on the training split of our :data:`bostondata` Dataset and evaluate it on the test split.
It returns an instance of :class:`~ml_tooling.result.Result` which we can then introspect further.

.. doctest::

    >>> result = linear.score_estimator(bostondata)
    >>> result
    <Result LinearRegression: {'r2': 0.68}>



Testing your model
~~~~~~~~~~~~~~~~~~

To test which estimator performs best, use the :meth:`~ml_tooling.baseclass.Model.test_estimator` method.
This method trains each estimator on the train split and evaluates the performance on the test split. It returns a new :class:`~ml_tooling.baseclass.Model` instance with the best-performing estimator
with the best estimator and a :class:`~ml_tooling.result.ResultGroup`.

.. doctest::

    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> best_model, results = Model.test_estimators(
    ...     bostondata,
    ...     [LinearRegression(), RandomForestRegressor(n_estimators=10, random_state=1337)],
    ...     metrics='r2')
    >>> results
    ResultGroup(results=[<Result RandomForestRegressor: {'r2': 0.83}>, <Result LinearRegression: {'r2': 0.68}>])

Training your model
~~~~~~~~~~~~~~~~~~~

When the best model has been found use :meth:`~ml_tooling.baseclass.Model.train_estimator` to train the model
on the full training set (not the training split).

.. note::

    This should be the last step before saving the model for production.

.. doctest::

    >>> linear.train_estimator(bostondata)
    <Model: LinearRegression>

Predicting with your model
~~~~~~~~~~~~~~~~~~~~~~~~~~

To make a prediction use the method :meth:`~ml_tooling.baseclass.Model.make_prediction`.
This will call the :meth:`~ml_tooling.data.Dataset.load_prediction_data` defined in your dataset.

.. doctest::

    >>> customer_id = 42
    >>> linear.make_prediction(bostondata, customer_id)
       Prediction
    0   25.203866

:meth:`~ml_tooling.baseclass.Model.make_prediction` also has a parameter :code:`proba` which will return the
underlying probabilities if working on a classification problem

Performing a gridsearch
~~~~~~~~~~~~~~~~~~~~~~~

To find the best hyperparameters for an estimator you can use
:meth:`~ml_tooling.baseclass.Model.gridsearch`, passing a dictionary of hyperparameters to try.

.. doctest::

    >>> best_estimator, results = linear.gridsearch(bostondata, { "normalize": [False, True] })
    >>> results
    ResultGroup(results=[<Result LinearRegression: {'r2': 0.72}>, <Result LinearRegression: {'r2': 0.72}>])

The input hyperparameters have a similar format to :class:`~sklearn.model_selection.GridSearchCV`, so if we are
gridsearching using a :class:`~sklearn.pipeline.Pipeline`, we can pass hyperparameters using the same syntax.

.. doctest::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> pipe = Pipeline([('scale', StandardScaler()), ('clf', LinearRegression())])
    >>> pipe_model = Model(pipe)
    >>> best_estimator, results = pipe_model.gridsearch(bostondata, { "clf__normalize": [False, True]})
    >>> results
    ResultGroup(results=[<Result LinearRegression: {'r2': 0.72}>, <Result LinearRegression: {'r2': 0.72}>])

Using the logging capability of Model :meth:`~ml_tooling.Model.log` method,
we can save each result to a yaml file.

.. doctest::

    >>> with linear.log("./bostondata_linear"):
    ...     best_estimator, results = linear.gridsearch(bostondata, { "normalize": [False, True] })

.. testcleanup::

    import shutil
    shutil.rmtree(linear.config.RUN_DIR.joinpath('bostondata_linear'))

This will generate a yaml file for each

.. code-block:: yaml

    created_time: 2019-10-31 17:32:08.233522
    estimator:
    - classname: LinearRegression
    module: sklearn.linear_model.base
    params:
        copy_X: true
        fit_intercept: true
        n_jobs: null
        normalize: true
    estimator_path: null
    git_hash: afa6def92a1e8a0ac571bec254129818bb337c49
    metrics:
        r2: 0.7160133196648374
    model_name: BostonData_LinearRegression
    versions:
        ml_tooling: 0.9.1
        pandas: 0.25.2
        sklearn: 0.21.3

Storage
-------

In order to store our estimators for later use or comparison, we use a
:class:`~ml_tooling.storage.Storage` class and pass it to :meth:`~ml_tooling.Model.save_estimator`.

.. testsetup::

    import pathlib
    pathlib.Path('./estimator_dir').mkdir(exist_ok=True)

.. doctest::

    >>> from ml_tooling.storage import FileStorage
    >>>
    >>> estimator_dir = './estimator_dir'
    >>> storage = FileStorage(estimator_dir)
    >>> estimator_path = linear.save_estimator(storage)
    >>> estimator_path.name # doctest: +SKIP
    'LinearRegression_2019-10-23_13:23:22.058684.pkl' # doctest: +SKIP

The model creates a filename for the model estimator based on the current date and time and the estimator name.

We can also load the model from a storage by specifying the filename to load in the Storage directory.

.. doctest::

    >>> loaded_linear = linear.load_estimator(storage, estimator_path.name)
    >>> loaded_linear
    <Model: LinearRegression>

.. testcleanup::

    import shutil
    shutil.rmtree(pathlib.Path('./estimator_dir'))

Saving an estimator ready for production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have a trained estimator ready to be saved for use in production on your filesystem.

.. code-block::

    >>> from ml_tooling.storage import FileStorage
    >>> from ml_tooling import Model
    ...
    >>> storage = FileStorage('./estimators/')
    >>> estimator = Filestorage.load('current_best_model.pkl')
    ...
    >>> model = Model(estimator)
    ...
    >>> model.save_estimator(storage, prod=True)

Now users of your model package can always find your estimator through
:meth:`~ml_tooling.Model.load_production_estimator` using the module name.

.. code-block::

    >>> model.load_production_estimator('your_module_name')


Configuration
-------------

To change the default configuration values, modify the :attr:`~Model.config` attributes directly:

.. doctest::

    >>> linear.config.RANDOM_STATE = 2

.. seealso::
    Refer to :ref:`config` for a list of available configuration options



Logging
-------

We also have the ability to log our experiments using the :meth:`Model.log` context manager.
The results will be saved in

.. doctest::

    >>> with linear.log('test_dir'):
    ...     linear.score_estimator(bostondata)
    <Result LinearRegression: {'r2': 0.68}>

.. testcleanup::

    import shutil
    shutil.rmtree(linear.config.RUN_DIR)

This will write a yaml file specifying attributes of the model, results, git-hash of the model
and other pertinent information.

.. seealso::

    Check out :meth:`Model.log` for more info on what is logged


Continue to :doc:`storage`

.. _scikit-learn: https://scikit-learn.org/stable/
