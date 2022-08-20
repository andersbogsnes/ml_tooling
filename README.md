# Model Tooling library
[![Build Status](https://github.com/andersbogsnes/ml_tooling/workflows/Integration/badge.svg)](https://github.com/andersbogsnes/ml_tooling/actions?workflow=Tests)
[![codecov](https://codecov.io/gh/andersbogsnes/ml_tooling/branch/main/graph/badge.svg)](https://codecov.io/gh/andersbogsnes/ml_tooling)
[![Python 3](https://img.shields.io/pypi/pyversions/ml_tooling.svg)](https://pyup.io/repos/github/andersbogsnes/ml_tooling/)
[![CodeFactor](https://www.codefactor.io/repository/github/andersbogsnes/ml_tooling/badge)](https://www.codefactor.io/repository/github/andersbogsnes/ml_tooling)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation
Use pip to install:
`pip install ml-tooling`
Or use conda
`conda install -c conda-forge ml_tooling`

## Test
We use `tox` for managing build and test environments, to install `tox` run:
`pip install tox`
And to run tests:
`tox -e py`

## Example usage
Define a class using ModelData and implement the two required methods.
Here we simply implement a linear regression on the Boston dataset using sklearn.datasets
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from ml_tooling import Model
from ml_tooling.data import Dataset

# Define a new data class
class BostonData(Dataset):
    def load_prediction_data(self, idx):
        x, _ = load_boston(return_X_y=True)
        return x[idx] # Return given observation

    def load_training_data(self):
        return load_boston(return_X_y=True)

# Instantiate a model with an estimator
linear_boston = Model(LinearRegression())

# Instantiate the data
data = BostonData()

# Split training and test data
data.create_train_test()

# Score the estimator yielding a Result object
result = linear_boston.score_estimator(data)

# Visualize the result
result.plot.prediction_error()

print(result)
<Result LinearRegression: {'r2': 0.68}>
```


## Links
* Documentation: https://ml-tooling.readthedocs.io
* Releases: https://pypi.org/project/ml_tooling/
* Code: https://github.com/andersbogsnes/ml_tooling
* Issue Tracker: https://github.com/andersbogsnes/ml_tooling/issues
