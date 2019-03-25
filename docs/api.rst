API
===

.. _api:

ModelData
--------------
Abstract BaseClass to inherit from.
Implements all the base functionality needed to create the wrapper

.. autoclass:: ml_tooling.baseclass.ModelData
    :members:

    .. automethod:: log(self, run_name)

    .. automethod:: setup_estimator()

.. include:: config.inc.rst

.. include:: result.inc.rst

.. include:: plots.inc.rst

.. include:: transformers.inc.rst