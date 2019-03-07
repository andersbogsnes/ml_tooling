.. _transformer:

Transformers
============

One great feature of scikit-learn is the concept of the
`Pipeline <https://scikit-learn.org/stable/modules/compose.html#pipeline>`_
alongside `Transformers <https://scikit-learn.org/stable/modules/preprocessing.html>`_


By default, scikit-learn's transformers will convert pandas dataframes to numpy arrays -
losing valuable column information in the process. We have implemented a number of transformers
that accept pandas dataframes and return pandas dataframes.
