# Model Tooling library
[![Build Status](https://travis-ci.org/andersbogsnes/ml_tooling.svg?branch=master)](https://travis-ci.org/andersbogsnes/ml_tooling)
[![codecov](https://codecov.io/gh/andersbogsnes/ml_tooling/branch/develop/graph/badge.svg)](https://codecov.io/gh/andersbogsnes/ml_tooling)
[![Python 3](https://pyup.io/repos/github/andersbogsnes/ml_tooling/python-3-shield.svg)](https://pyup.io/repos/github/andersbogsnes/ml_tooling/)
[![Updates](https://pyup.io/repos/github/andersbogsnes/ml_tooling/shield.svg)](https://pyup.io/repos/github/andersbogsnes/ml_tooling/)

# Installation
Use pip to install: 
`pip install ml-tooling`

# Contents
* Transformers
    * A library of transformers for use with Scikit-learn pipelines

* Model base classes
    * Production baseclasses for subclassing - guarantees interface for use in API

* Plotting functions
    * Functions for producing nice, commonly used plots such as roc_curves and confusion matrices 

# BaseClassModel
A base Class for defining your model. 
Your subclass must define two methods:
 
- `get_prediction_data()`
 
    Function that, given an input, fetches corresponding features. Used for predicting an unseen observation
 
- `get_training_data()`
    
    Function that retrieves all training data. Used for training and evaluating the model


## Example usage
Define a class using BaseClassModel and implement the two required methods.
Here we simply implement a linear regression on the Boston dataset using sklearn.datasets
```python
from ml_tooling import BaseClassModel
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, LassoLars

# Define a new class

class BostonModel(BaseClassModel):
    def get_prediction_data(self, idx):
        x, _ = load_boston(return_X_y=True)
        return x[idx] # Return given observation
        
    def get_training_data(self):
        return load_boston(return_X_y=True)
    
# Use our new class to implement a given model - any sklearn compatible estimator
linear_boston = BostonModel(LinearRegression())

results = linear_boston.score_model()

# Visualize results
results.plot.residuals()
results.plot.prediction_error()

# Save our model
linear_boston.save_model()

# Recreate model
BostonModel.load_model('.')

# Train Different models and get the best performing
models_to_try = [LinearRegression(), Ridge(), LassoLars()]

# best_model will be BostonModel instantiated with the highest scoring model. all_results is a list of all results 
best_model, alL_results = BostonModel.test_models(models_to_try, metric='neg_mean_squared_error')
alL_results.to_dataframe(params=False)

```
## Configuration
ml_tooling implements a number of default settings for modelling.

You can see the configuration by accessing the .config attribute.
This config can be changed by simply modifying the relevant config.
```python
BostonModel.config.RANDOM_STATE = 2
```

## Config Options
### VERBOSITY
Defaults to 0. The level of verbosity sklearn outputs
### CLASSIFIER_METRIC
Defaults to 'accuracy'. The metric to use in classifier problems
### REGRESSION_METRIC
Defaults to 'r2'. The metric to use in regression problems
### CROSS_VALIDATION 
Defaults to 10. Number of cross validation iterations to use
### STYLE_SHEET
Defaults to built-in Alm Brand styling. Path to a matplotlib stylesheet to use in plots
### N_JOBS 
Defaults to -1. Number of cores to use in multiprocessing situations.
### TEST_SIZE 
Defaults to 0.25. Percentage of data taken out as test data.
### RANDOM_STATE 
Defaults to 42. Random state seed to use.
### RUN_DIR 
Defaults to './runs'. Folder to store run logging files 
### MODEL_DIR 
Defaults to './models'. Folder to store pickled models
### LOG 
Defaults to False. Toggles whether or not to log to file. Set to True 
if you want every run to be logged, otherwise use the `.log` 
context manager to toggle for a single run.

## Methods
The BaseClass implements a number of useful methods

### `save_model(path=None)`
Saves the model as a binary file. Defaults to current working directory, 
with a filename of `<class_name>_<model_name>_<git_hash>.pkl`

### `load_model(path)`
Instantiates the class with a joblib pickled model.
If no path is given, searches path for the newest file that matches the pattern 
   
### `score_model(metric='accuracy', cv=False)`
Loads all training data and trains the model on it, using a train_test split.
Returns a Result object containing all result parameters
Defaults to non-cross-validated scoring. If you want to cross-validate, pass number of folds to cv

### `train_model()`
Loads all training data and trains the model on all data. 
Typically used as the last step when model tuning is complete.
Sets .result attribute to None

   
### `make_prediction(*args)`
Makes a prediction given an input. For example a customer number. 
Passed to the implemented `get_prediction_data()` method and calls `.predict()` on the estimator
   

### `test_models([model1, model2], metric='accuracy')`
Runs `score_model()` on each model, saving the result.
Returns the best model as well as a ResultGroup of all results

### `gridsearch(param_grid)`
Runs a gridsearch on the model with the passed in parameter grid.
The function will ensure that it works inside a pipeline as well.

### `setup_model()`
To be implemented by the user - setup_model is a classmethod which loads up an untrained model.
Typically this would setup a pipeline and the selected model for easy training

Returning to our previous example of the BostonModel, let us implement a setup_model method
```python
from ml_tooling import BaseClassModel
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd

class BostonModel(BaseClassModel):
    def get_prediction_data(self, idx):
        data = load_boston()
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        return df.iloc[idx] # Return given observation
        
    def get_training_data(self):
        data = load_boston()
        return pd.DataFrame(data=data.data, columns=data.feature_names), data.target
    
    @classmethod
    def setup_model(cls):
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearRegression())
        ])
        return cls(pipeline)
```

Given this extra setup, it becomes easy to load the untrained model to train it:
```python
model = BostonModel.setup_model()
model.train_model()
```



### `log(log_dir)`
`log()` is a context manager that lets you turn on logging for any scoring methods that follow. 
You can pass a log_dir to specify a subfolder to store the model in. The output is a yaml
file recording model parameters, package version numbers, metrics and other useful information

Usage example:
```python
model = BostonModel.setup_model()

with model.log('score'):
    model.score_model()

``` 

This will save the results of `model.score_model()` to `runs/score/`

## Visualizing results
When a model is trained, it returns a Result object. 
That object has number of visualization options depending on the type of model:

Any visualizer listed here also has a functional counterpart in `ml_tooling.plots`.
E.g if you want to use the function for plotting a confusion matrix without using 
the ml_tooling BaseClassModel approach, you can instead do 
`from ml_tooling.plots import plot_confusion_matrix`

These functional counterparts all mirror sklearn metrics api, taking y_target and y_pred
as arguments

```python
from ml_tooling.plots import plot_confusion_matrix
import numpy as np

y_true = np.array([1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 0])
plot_confusion_matrix(y_true, y_pred)
```

### Classifiers
   
- `roc_curve(**kwargs)`:<br />  Visualize a ROC curve for a classification model. Model must implement a `predict_proba` method. Any kwargs are passed onto matplotlib. <br /> <br />
- `confusion_matrix(normalized = True, **kwargs)`: 
Visualize a confusion matrix for a classification model. `normalized` determines whether or not to normalize annotated class counts. Any kwargs are passed onto matplotlib.  <br /> <br />
- `feature_importance(samples, values = True,  top_n = None, bottom_n = None, seed=1337, **kwargs)`:<br /> Calculates each features imporatance with permutation. Importance is measured in drop in model metric. `samples` determines the number of samples to use and must be set. <br /> <br />
If `samples=None` the original data set is used which is not recommended for small data sets. <br /> <br />
If `samples` is a `float` between 0 and 1 a new smaller data set is made from resampling with replacement form the original data set. This is not recommended for small data sets but could be suitable for very large data sets. <br /> <br />
If  `samples` is set to an `int` a new  data set of size `samples` is made from resampling with replacement form the original data. Recommended for small data sets to ensure stable estimates of feature importance.  <br /> <br />
If `top_n` is an `integer`, return `top_n` features and if `top_n` is a `float` between `(0, 1)`, return `top_n` percent features. <br /> <br /> If `bottom_n` is an `integer`, return `bottom_n` features and if `bottom_n` is a `float` between `(0, 1)`, return `bottom_n` percent features. <br /> <br />
- `lift_curve(**kwargs)`: <br />
Visualize a Lift Curve for a classification model. Model must implement a `predict_proba` method. Any kwargs are passed onto matplotlib.

### Regressors
   
- `prediction_error(**kwargs)`:<br /> 
Visualizes prediction error of a regression model. Any kwargs are passed onto matplotlib.
 <br /> <br />
- `residuals(**kwargs)`: <br /> 
Visualizes residuals of a regression model. Any kwargs are passed onto matplotlib.
 <br /> <br />
- `feature_importance(samples, values = True,  top_n = None, bottom_n = None, seed=1337, **kwargs)`:<br /> Calculates each features imporatance with permutation. Importance is measured in drop in model metric. `samples` determines the number of samples to use and must be set. <br /> <br />
If `samples=None` the original data set is used which is not recommended for small data sets. <br /> <br />
If `samples` is a `float` between 0 and 1 a new smaller data set is made from resampling with replacement form the original data set. This is not recommended for small data sets but could be suitable for very large data sets. <br /> <br />
If  `samples` is set to an `int` a new  data set of size `samples` is made from resampling with replacement form the original data. Recommended for small data sets to ensure stable estimates of feature importance.  <br /> <br />
If `top_n` is an `integer`, return `top_n` features and if `top_n` is a `float` between `(0, 1)`, return `top_n` percent features. <br /> <br /> If `bottom_n` is an `integer`, return `bottom_n` features and if `bottom_n` is a `float` between `(0, 1)`, return `bottom_n` percent features. <br /> <br />


# Transformers
The library also provides a number of transformers for working with DataFrames in a pipeline
## Select
A column selector - Provide a list of columns to be passed on in the pipeline
### Example
```python
from ml_tooling.transformers import Select
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "status": ["OK", "Error", "OK", "Error"],
    "sales": [2000, 3000, 4000, 5000] 

})

select = Select(['id', 'status'])
select.fit_transform(df)
```
```
Out[1]: 
   id status
0   1     OK
1   2  Error
2   3     OK
3   4  Error
```

## FillNA
Fills NA values with given value or strategy. Either a value or a strategy has to be supplied.
### Example for value
```python
from ml_tooling.transformers import FillNA
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "status": ["OK", "Error", "OK", "Error"],
    "sales": [2000, 3000, 4000, np.nan] 

})

fill_na = FillNA(value = 0)
fill_na.fit_transform(df)
```
```
Out[1]: 
   id status   sales
0   1     OK  2000.0
1   2  Error  3000.0
2   3     OK  4000.0
3   4  Error     0.0

```

### Example for strategy
The built-in strategies are 'mean', 'median', 'most_freq', 'max' and 'min. An example of 'mean' would be:
```python

fill_na = FillNA(value='mean')
fill_na.fit_transform(df)
```
```
Out[1]: 
   id status   sales
0   1     OK  2000.0
1   2  Error  3000.0
2   3     OK  4000.0
3   4  Error  3000.0

```

## ToCategorical
Performs one-hot encoding of categorical values through pd.Categorical. 
All categorical values not found in training data will be set to 0 

### Example
```python
from ml_tooling.transformers import ToCategorical
import pandas as pd

df = pd.DataFrame({
    "status": ["OK", "Error", "OK", "Error"] 

})

onehot = ToCategorical()
onehot.fit_transform(df)
```
```
Out[1]: 
   status_Error  status_OK
0             0          1
1             1          0
2             0          1
3             1          0
```

## FuncTransformer
Applies a given function to each column

### Example
```python
from ml_tooling.transformers import FuncTransformer
import pandas as pd

df = pd.DataFrame({
    "status": ["OK", "Error", "OK", "Error"]
})

uppercase = FuncTransformer(lambda x: x.str.upper())
uppercase.fit_transform(df)
```
```
Out[1]: 
  status
0     OK
1  ERROR
2     OK
3  ERROR
```

Keyword arguments can be supplied to the function. 
```python
from ml_tooling.transformers import FuncTransformer
import pandas as pd

def custom_func(input, word1, word2):
   result = ""
   if input == "OK":
      result = word1
   elif input == "Error":
      result = word2
   return result

def wrapper(df, word1, word2):
   return df.apply(custom_func,args=(word1,word2))

df = pd.DataFrame({
    "status": ["OK", "Error", "OK", "Error"]
})

kwargs = {'word1': 'Okay','word2': 'Fail'}
wordchange = FuncTransformer(wrapper,**kwargs)
wordchange.fit_transform(df)
```

```
Out[2]: 
  status
0   Okay
1   Fail
2   Okay
3   Fail
```


## Binner
Bins numerical data into supplied bins

### Example
```python
from ml_tooling.transformers import Binner
import pandas as pd

df = pd.DataFrame({
    "sales": [1500, 2000, 2250, 7830]
})

binned = Binner(bins=[0, 1000, 2000, 8000])
binned.fit_transform(df)
```
```
Out[1]: 
          sales
0  (1000, 2000]
1  (1000, 2000]
2  (2000, 8000]
3  (2000, 8000]
```

## Renamer
Renames columns to be equal to the passed list - must be in order

### Example
```python
from ml_tooling.transformers import Renamer
import pandas as pd

df = pd.DataFrame({
    "Total Sales": [1500, 2000, 2250, 7830]
})

rename = Renamer(['sales'])
rename.fit_transform(df)
```

```
Out[1]: 
   sales
0   1500
1   2000
2   2250
3   7830
```

## DateEncoder
Adds year, month, day, week columns based on a datefield. Each date type can be toggled in the initializer

### Example
```python
from ml_tooling.transformers import DateEncoder
import pandas as pd

df = pd.DataFrame({
    "sales_date": [pd.to_datetime('2018-01-01'), pd.to_datetime('2018-02-02')]
})

dates = DateEncoder(week=False)
dates.fit_transform(df)
```

```
Out[1]: 
   sales_date_day  sales_date_month  sales_date_year
0               1                 1             2018
1               2                 2             2018
```

## FreqFeature
Converts a column into a normalized frequencies

### Example
```python
from ml_tooling.transformers import FreqFeature
import pandas as pd

df = pd.DataFrame({
    "sales_category": ['Sale', 'Sale', 'Not Sale']
})

freq = FreqFeature()
freq.fit_transform(df)
```
```
Out[1]: 
   sales_category
0        0.666667
1        0.666667
2        0.333333
```

## DFFeatureUnion
A FeatureUnion equivalent for DataFrames. Concatenates the result of multiple transformers

### Example
```python
from ml_tooling.transformers import FreqFeature, Binner, Select, DFFeatureUnion
from sklearn.pipeline import Pipeline
import pandas as pd


df = pd.DataFrame({
    "sales_category": ['Sale', 'Sale', 'Not Sale', 'Not Sale'],
    "sales": [1500, 2000, 2250, 7830]
})


freq = Pipeline([
    ('select', Select('sales_category')), 
    ('freq', FreqFeature())
])

binned = Pipeline([
    ('select', Select('sales')), 
    ('bin', Binner(bins=[0, 1000, 2000, 8000]))
    ])


union = DFFeatureUnion([
    ('sales_category', freq), 
    ('sales', binned)
])
union.fit_transform(df)
```
```
Out[1]: 
   sales_category         sales
0             0.5  (1000, 2000]
1             0.5  (1000, 2000]
2             0.5  (2000, 8000]
3             0.5  (2000, 8000]
```

## DFRowFunc
Row-wise operation on Pandas DataFrame. Strategy can either be one of the predefined or a callable. If some elements in the row are NaN these elements are ignored for the built-in strategies.

### Example
```python
from ml_tooling.transformers import DFRowFunc
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "number_1": [1, np.nan, 3, 4],
    "number_2": [1, 3, 2, 4]

})

rowfunc = DFRowFunc(strategy = 'sum')
rowfunc.fit_transform(df)
```
```
Out[1]: 
         0
0        2
1        3
2        5
3        8
```

The built-in strategies are 'sum', 'min' and 'max'. A strategy can also be a callable:

```python

rowfunc = DFRowFunc(strategy = np.mean)
rowfunc.fit_transform(df)
```
```
Out[1]: 
         0
0        1
1        3
2        2.5
3        4

```


## Binarize
Transformer which returns 1 if equal to given value else 0.

### Example
```python
from ml_tooling.transformers import Binarize
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "number_1": [1, np.nan, 3, 4],
    "number_2": [1, 3, 2, 4]

})

binarize = Binarize(value = 3)
binarize.fit_transform(df)
```
```
Out[1]: 
         number_1    number_2
0               0           0
1               1           0
2               0           1
3               0           0
```
