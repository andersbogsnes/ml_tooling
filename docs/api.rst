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


Config
------
All configuration options available

.. module:: ml_tooling.config
.. autoclass:: DefaultConfig

    :attr:`VERBOSITY` = 0

    The level of verbosity from output


    :attr:`CLASSIFIER_METRIC` = 'accuracy'

    Default metric for classifiers

    :attr:`REGRESSION_METRIC` = 'r2'

    Default metric for regressions

    :attr:`CROSS_VALIDATION` = 10

    Default Number of cross validation folds to use

    :attr:`STYLE_SHEET` = 'almbrand.mplstyle'

    Default style sheet to use for plots

    :attr:`N_JOBS` = -1

    Default number of cores to use when doing multiprocessing.
    -1 means use all available

    :attr:`TEST_SIZE` = 0.25

    Default test size percentage. How much data is witheld to use for testing

    :attr:`RANDOM_STATE` = 42

    Default random state seed for all functions involving randomness

    :attr:`RUN_DIR` = './runs'

    Default folder to store run logging files

    :attr:`MODEL_DIR` = './models'

    Default folder to store pickled models in

    :attr:`LOG` = False

    Toggles whether or not to log runs to a file. Set to True if you
    want every run to be logged, else use the :meth:`~.BaseClassModel.log`
    context manager
