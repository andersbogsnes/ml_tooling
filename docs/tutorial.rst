.. _tutorial:


We will be using the `Iris dataset <https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_ for this tutorial.
First we create the dataset we will be working with.

Tutorial
==========


.. doctest::

    >>> from sklearn.datasets import load_iris
    >>> from ml_tooling.data import Dataset
    >>> import pandas as pd
    >>> import numpy as np

.. doctest::

    >>> class IrisData(Dataset):
    ...     def load_training_data(self):
    ...         data = load_iris()
    ...         target = np.where(data.target == 1, 1, 0)
    ...         return pd.DataFrame(data=data.data, columns=data.feature_names), target
    ...
    ...     def load_prediction_data(self, idx):
    ...         X, y = self.load_training_data()
    ...         return X.loc[idx, :].to_frame().T
    >>>
    >>> data = IrisData()
    >>> data.create_train_test()
    <IrisData - Dataset>

.. doctest::

    >>> from ml_tooling import Model
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> lr_clf = Model(LogisticRegression())
    >>>
    >>> lr_clf.score_estimator(data, metrics='accuracy')
    <Result LogisticRegression: {'accuracy': 0.74}>

We have a few more model we'd like to try out and see what performs best.
We can include a random classifier to

.. doctest::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf_clf = Model(RandomForestClassifier(n_estimators=10, random_state=42))
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb_clf = Model(GradientBoostingClassifier())


We can test the estimators in tandem with cross validation like so:

.. doctest::

    >>> estimators = [lr_clf.estimator, rf_clf.estimator, gb_clf.estimator]
    >>> best_model, results = Model.test_estimators(data, estimators, metrics='accuracy', cv=10)
    >>> results
    ResultGroup(results=[<Result RandomForestClassifier: {'accuracy': 0.95}>, <Result GradientBoostingClassifier: {'accuracy': 0.93}>, <Result LogisticRegression: {'accuracy': 0.66}>])



.. doctest::

    >>> with rf_clf.log('./gridsearch'):
    ...     best_model, results = rf_clf.gridsearch(data, {"max_depth": [3, 5, 10, 15]})
    >>>
    >>> results
    ResultGroup(results=[<Result RandomForestClassifier: {'accuracy': 0.95}>, <Result RandomForestClassifier: {'accuracy': 0.95}>, <Result RandomForestClassifier: {'accuracy': 0.95}>, <Result RandomForestClassifier: {'accuracy': 0.95}>])

.. testcleanup::

    import shutil
    shutil.rmtree(best_model.config.RUN_DIR.joinpath('gridsearch'))

We do the gridsearch in a .log() context manager so we can inspect the gridsearched models, and recreate them later if we need to.

.. code-block::

    >>> best_model.result.plot.feature_importance()

.. plot::

    >>> from sklearn.datasets import load_iris
    >>> from ml_tooling.data import Dataset
    >>> import pandas as pd
    >>> import numpy as np
    >>> class IrisData(Dataset):
    ...     def load_training_data(self):
    ...         data = load_iris()
    ...         target = np.where(data.target == 1, 1, 0)
    ...         return pd.DataFrame(data=data.data, columns=data.feature_names), target
    ...
    ...     def load_prediction_data(self, idx):
    ...         X, y = self.load_training_data()
    ...         return X.loc[idx, :].to_frame().T
    >>>
    >>> data = IrisData()
    >>> data.create_train_test()
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf_clf = Model(RandomForestClassifier(n_estimators=10, random_state=42))
    >>> best_model, results = rf_clf.gridsearch(data, {"max_depth": [3, 5, 10, 15]})
    >>> best_model.result.plot.feature_importance()

.. doctest::

    >>> from ml_tooling.storage import FileStorage
    >>>
    >>> storage = FileStorage('./estimators)
    >>> best_model.save_estimator(storage)

.. testcleanup::

    import shutil
    import pathlib
    shutil.rmtree(pathlib.Path('./estimators'))
