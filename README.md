# Model Utility library for Alm Brand
[![Build Status](https://travis-ci.org/andersbogsnes/ml_utils.svg?branch=master)](https://travis-ci.org/andersbogsnes/ml_utils)
[![Coverage Status](https://coveralls.io/repos/github/andersbogsnes/ml_utils/badge.svg?branch=master)](https://coveralls.io/github/andersbogsnes/ml_utils?branch=master)
## Contents
* Transformers
    A library of transformers for use with Scikit-learn pipelines
* Model base classes
    Production baseclasses for subclassing - guarantees interface for use in API
        
### BaseClassModel
A baseclass for defining your model. 
Must define two methods:
 
 - `get_prediction_data()`
 
    Function that, given an input, fetches corresponding features. Used for predicting an unseen observation
 
 - `get_training_data`
    
    Function that retrieves all training data. Used for training and evaluating the model


### Example usage
Define a class using BaseClassModel and implement the two required methods.
Here we simply implement a linear regression on the Boston dataset using sklearn.datasets
```python
from ml_utils import BaseClassModel
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# Define a new class

class BostonModel(BaseClassModel):
    def get_prediction_data(self, idx):
        x, _ = load_boston(return_X_y=True)
        return x[idx] # Return given observation
        
    def get_training_data(self):
        return load_boston(return_X_y=True)
    
# Use our new class to implement a given model - any sklearn compatible estimator
linear_boston = BostonModel(LinearRegression())

results = linear_boston.train_model()

# Visualize results
results.plot.residuals()
results.plot.prediction_error()

# Save our model
linear_boston.save_model('.')

# Recreate model
BostonModel.load_model('./LinearRegression')

```

Implements a number of useful methods
#### `save_model()`
   Saves the model as a binary file
   
### `load_model` 
   Instantiates the class with a joblib pickled model
#### `train_model`
   Loads all training data and trains the model on it, using a train_test split.
   Returns a Result object containing all result parameters
   
