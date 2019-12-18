.. currentmodule:: ml_tooling.result.viz
.. _plotting:

Plotting
========

Available Base plots
--------------------

First we define a :class:`~ml_tooling.data.Dataset` like we have done in previous examples.
When we score the estimator by calling :meth:`~ml_tooling.Model.score_estimator`, we get a
:class:`~ml_tooling.result.Result` back, which contains a number of handy plotting features.

To use the visualizations, access them using the `.plot` accessor on the
:class:`~ml_tooling.result.Result` object:


.. code-block::

    >>> result.plot.feature_importance()

.. plot::

    >>> result.plot.feature_importance()


.. code-block:: python

    >>> result.plot.residuals()

.. plot::

    >>> result.plot.residuals()

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
    Uses random permutation to calculate feature importance by randomly permuting each column
    and measuring the difference in the model metric against the baseline.

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

Continue to :doc:`transformers`
