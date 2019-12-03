.. _storage:

Storage
=======

ML Tooling provides different backends for storing trained models. Currently, we support local file storage, as well as
Artifactory based storage.


ArtifactoryStorage
------------------

If you want to use Artifactory as a backend, first install the optional dependencies by running
:code:`pip install ml_tooling['artifactory']`.


Saving and loading an estimator from Artifactory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    >>> from ml_tooling.storage import ArtifactoryStorage
    >>> from ml_tooling import Model
    ...
    >>> artifactory_url = "http://artifactory.com/artifactory"
    >>> artifactory_repo = "advanced-analytics/dev/myfolder"
    >>>
    >>> storage = ArtifactoryStorage(artifactory_url, artifactory_repo)
    ...
    >>> estimators_in_myfolder = storage.get_list()
    >>> estimators_in_myfolder
    [ArtifactoryPath('http://artifactory.com/artifactory/advanced-analytics/dev/LinearRegression_2019-10-16_15:10:34.290209.pkl'), ArtifactoryPath('http://artifactory.com/artifactory/advanced-analytics/dev/LinearRegression_2019-10-16_15:14:02.114818.pkl')]
    >>> my_estimator = storage.load(estimators_in_myfolder[0])
    >>> my_estimator
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    ...
    >>> from ml_tooling import Model
    >>> model = Model(my_estimator)
    >>> model
    <Model: LinearRegression>
    >>> # ...
    >>> # make some adjustments to the model
    ...
    >>> # now save the estimator, ideally using the Model instance
    >>> model.save_estimator(storage)
    [14:34:08] - Saved estimator to http://artifactory-singlep.com/artifactory/advanced-analytics/dev/LinearRegression_2019-10-30_14:34:08.116648.pkl
    ArtifactoryPath('http://artifactory.com/artifactory/advanced-analytics/dev/LinearRegression_2019-10-30_14:34:08.116648.pkl')
    >>> # if you want more control over folders/name destinations use the Storage directly
    >>> new_estimator_path = "my_new_folder/LinearRegression.pkl"
    >>> storage.save(model.estimator, new_estimator_path)
    ArtifactoryPath('http://artifactory.com/artifactory/advanced-analytics/dev/my_new_folder/LinearRegression.pkl')

For more information see :class:`~ml_tooling.storage.ArtifactoryStorage`.

FileStorage
-----------

Saving and loading an estimator from the file system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    >>> from ml_tooling.storage import FileStorage
    >>> from ml_tooling import Model
    ...
    >>> my_folder = "./estimators"
    ...
    >>> storage = FileStorage(my_folder)
    ...
    >>> estimators_in_myfolder = storage.get_list()
    >>> my_estimator = storage.load(estimators_in_myfolder[0])
    ...
    >>> new_model_name = "my_new_folder/LinearRegression.pkl"
    >>> storage.save(model.estimator)
    PosixPath('estimators/my_new_folder/LinearRegression.pkl')

See :class:`~ml_tooling.storage.FileStorage` for more information

Continue to :doc:`plotting`
