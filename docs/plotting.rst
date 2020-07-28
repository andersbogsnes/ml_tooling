.. currentmodule:: ml_tooling.plots.viz
.. _plotting:

Plotting
========

Available Base plots
--------------------

First we define a :class:`~ml_tooling.data.Dataset` like we have done in Quickstart and Tutorial.
When we score the estimator by calling :meth:`~ml_tooling.baseclass.Model.score_estimator`, we get a
:class:`~ml_tooling.result.Result` back, which contains a number of handy plotting features.

To use the visualizations, access them using the `.plot` accessor on the
:class:`~ml_tooling.result.Result` object:


.. code-block::

    >>> result.plot.feature_importance()

.. plot::

    >>> result.plot.feature_importance()

.. code-block::

    >>> result.plot.permutation_importance()

.. plot::

    >>> result.plot.permutation_importance()

.. code-block:: python

    >>> result.plot.residuals()

.. plot::

    >>> result.plot.residuals()

.. code-block:: python

    >>> result.plot.learning_curve()

.. plot::

    >>> result.plot.learning_curve()

Any visualization listed here also has a functional counterpart in :mod:`ml_tooling.plots`.
E.g if you want to use the function for plotting a confusion matrix without using
the :class:`~ml_tooling.result.Result` class

.. doctest:: python

    >>> from ml_tooling.plots import plot_confusion_matrix

These functional counterparts all mirror the sklearn metrics api, taking y_target and y_pred
as arguments:

.. code-block:: python

    >>> from ml_tooling.plots import plot_confusion_matrix
    >>> import numpy as np
    >>>
    >>> y_true = np.array([1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 0, 0])
    >>> plot_confusion_matrix(y_true, y_pred)

.. plot::

    >>> from ml_tooling.plots import plot_confusion_matrix
    >>> import numpy as np
    >>>
    >>> y_true = np.array([1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 0, 0])
    >>> plot_confusion_matrix(y_true, y_pred)


Available Base plots
--------------------
- :meth:`~RegressionVisualize.feature_importance`
    Uses the estimator's learned coefficients or learned feature importance in the case of RandomForest
    to plot the relative importance of each feature. Note that for most usecases, permutation importance is
    going to be more accurate, but is also more computationally expensive

- :meth:`~RegressionVisualize.permutation_importance`
    Uses random permutation to calculate feature importance by randomly permuting each column
    and measuring the difference in the model metric against the baseline.

- :meth:`~RegressionVisualize.learning_curve`
    Draws a learning curve, showing how number of training examples affects model performance.
    Can also be used to diagnose overfitting and underfitting by examining training and validation
    set performance

- :meth:`~RegressionVisualize.validation_curve`
    Visualizes the impact of a given hyperparameter on the model metric by plotting a range
    of different hyperparameter values

Available Classifier plots
--------------------------
- :meth:`~ClassificationVisualize.roc_curve`
    Visualize a ROC curve for a classification model.
    Shows the relationship between the True Positive Rate and the False Positive Rate.

- :meth:`~ClassificationVisualize.confusion_matrix`:
    Visualize a confusion matrix for a classification model.
    Shows the distribution of predicted labels vs actual labels

- :meth:`~ClassificationVisualize.lift_curve`
    Visualizes how much of the target class we capture by setting different thresholds for probability

- :meth:`~ClassificationVisualize.pr_curve`
    Visualize a Precision-Recall curve for a classification estimator. Estimator must implement a `predict_proba` method.

Available Regression Plots
--------------------------
- :meth:`~RegressionVisualize.prediction_error`:
    Visualizes prediction error of a regression model. Shows how far away each prediction is
    from the correct prediction for that point

- :meth:`~RegressionVisualize.residuals`:
    Visualizes residuals of a regression model. Shows the distribution of noise that couldn't be
    fitted.

Data Plotting
=============
.. currentmodule:: ml_tooling.data.viz

:class:`~ml_tooling.data.Dataset` also define plotting methods under the `.plot` accessor.

These plots are intended to help perform exploratory data analysis to inform the choices of
preprocessing and models

These plot methods are used the same way as the result plots

.. code-block::

    >>> bostondata.plot.target_correlation()

.. plot::

    >>> bostondata.plot.target_correlation()

Optionally, you can pass a preprocessing :class:`~sklearn.pipeline.Pipeline` to the plotter to preprocess the data
before plotting. This can be useful if you want to check that the preprocessing is handling all the NaNs, or
if you want to visualize computed columns.

.. code-block::

    >>> from ml_tooling.transformers import DFStandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> feature_pipeline = Pipeline([("scaler", DFStandardScaler())])
    >>> bostondata.plot.target_correlation(feature_pipeline=feature_pipeline)

Available Data Plots
--------------------

- :meth:`~ml_tooling.plots.viz.data_viz.DataVisualize.target_correlation`:
    Visualizes the correlations between each feature and the target variable.
    The size of the correlation can indicate important features, but can also
    hint at data leakage if the correlation is too strong.

- :meth:`~ml_tooling.plots.viz.data_viz.DataVisualize.missing_data`:
    Visualizes percentage of missing data for each column in the dataset. If no columns have missing data, will
    simply show an empty plot.

Continue to :doc:`transformers`
