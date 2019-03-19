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

.. include:: config.inc.rst

.. include:: result.inc.rst

.. include:: plots.inc.rst