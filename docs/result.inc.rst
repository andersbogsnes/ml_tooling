.. _result:

Result
------
Result class to work with results from scoring a model

.. autoclass:: ml_tooling.result.Result
    :members:

ResultGroup
-----------
A container of Results - some methods in ML Tooling return multiple results, which will be grouped into a
ResultGroup. A ResultGroup is sorted by the Result metric and proxies attributes to the best result

.. autoclass:: ml_tooling.result.ResultGroup
    :members:

Classification Result Visualizations
------------------------------------

.. autoclass:: ml_tooling.plots.viz.ClassificationVisualize
    :members:
    :inherited-members:

Regression Result Visualizations
--------------------------------

.. autoclass:: ml_tooling.plots.viz.RegressionVisualize
    :members:
    :inherited-members:
