# Model Utility library for Alm Brand
[![Build Status](https://travis-ci.org/andersbogsnes/ml_utils.svg?branch=master)](https://travis-ci.org/andersbogsnes/ml_utils)
[![Coverage Status](https://coveralls.io/repos/github/andersbogsnes/ml_utils/badge.svg?branch=master)](https://coveralls.io/github/andersbogsnes/ml_utils?branch=master)

# Installation
Use pip to install
`pip install git+https://git@github.com/andersbogsnes/ml_utils.git`

# Contents
* Transformers
    - A library of transformers for use with Scikit-learn pipelines
* Model base classes
    - Production baseclasses for subclassing - guarantees interface for use in API
        
## BaseClassModel
A base Class for defining your model. 
Your subclass must define two methods:
 
- `get_prediction_data()`
 
    Function that, given an input, fetches corresponding features. Used for predicting an unseen observation
 
- `get_training_data()`
    
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

The BaseClass implements a number of useful methods

#### `save_model()`
Saves the model as a binary file
   
### `load_model()` 
Instantiates the class with a joblib pickled model
   
#### `test_model()`
Loads all training data and trains the model on it, using a train_test split.
Returns a Result object containing all result parameters

### `train_model()`
Loads all training data and trains the model on all data. 
Typically used as the last step when model tuning is complete

### `set_config()`
Set configuration options - existing configuration options can be seen using the `.config` property
   
### `make_prediction(*args)`
Makes a prediction given an input. For example a customer number. 
Passed to the implemented `get_prediction_data()` method and calls `.predict()` on the estimator
   

## Visualizing results
When a model is trained, it returns a Result object. 
That object has number of visualization options depending on the type of model:
   
### Classifiers
   
- `roc_curve()`
- `confusion_matrix()`
   
### Regressors
   
- `prediction_error()`
- `residuals()`

# Transformers
The library also provides a number of transformers for working with DataFrames in a pipeline
### Select
A column selector - Provide a list of columns to be passed on in the pipeline
#### Example
```python
from ml_utils.transformers import Select
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "status": ["OK", "Error", "OK", "Error"],
    "sales": [2000, 3000, 4000, 5000] 

})

select = Select(['id', 'status'])
select.fit_transform(df)
```

### FillNA
Fills NA values with instantiated value - passed to df.fillna()
#### Example
```python
from ml_utils.transformers import FillNA
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "status": ["OK", "Error", "OK", "Error"],
    "sales": [2000, 3000, 4000, np.nan] 

})

fill_na = FillNA(0)
fill_na.fit_transform(df)
```

### ToCategorical
Performs one-hot encoding of categorical values through pd.Categorical. 
All categorical values not found in training data will be set to 0 

#### Example
```python
from ml_utils.transformers import ToCategorical
import pandas as pd

df = pd.DataFrame({
    "status": ["OK", "Error", "OK", "Error"] 

})

onehot = ToCategorical()
onehot.fit_transform(df)
```

### FuncTransformer
Applies a given function to each column

#### Example
```python
from ml_utils.transformers import FuncTransformer
import pandas as pd

df = pd.DataFrame({
    "status": ["OK", "Error", "OK", "Error"]
})

uppercase = FuncTransformer(lambda x: x.str.upper)
uppercase.fit_transform(df)
```

### Binner
Bins numerical data into supplied bins

#### Example
```python
from ml_utils.transformers import Binner
import pandas as pd

df = pd.DataFrame({
    "sales": [1500, 2000, 2250, 7830]
})

binned = Binner(bins=[0, 1000, 2000, 8000])
binned.fit_transform(df)
```

### Renamer
Renames columns to be equal to the passed list - must be in order

#### Example
```python
from ml_utils.transformers import Renamer
import pandas as pd

df = pd.DataFrame({
    "Total Sales": [1500, 2000, 2250, 7830]
})

rename = Renamer(['sales'])
rename.fit_transform(df)
```


### DateEncoder
Adds year, month, day, week columns based on a datefield. Each date type can be toggled in the initializer

```python
from ml_utils.transformers import DateEncoder
import pandas as pd

df = pd.DataFrame({
    "sales_date": [pd.to_datetime('2018-01-01'), pd.to_datetime('2018-02-02')]
})

dates = DateEncoder(week=False)
dates.fit_transform(df)
```

### FreqFeature
Converts a column into a normalized frequencies

```python
from ml_utils.transformers import FreqFeature
import pandas as pd

df = pd.DataFrame({
    "sales_category": ['Sale', 'Sale', 'Not Sale']
})

freq = FreqFeature()
freq.fit_transform(df)
```

### DFFeatureUnion
A FeatureUnion equivalent for DataFrames. Concatenates the result of multiple transformers

```python
from ml_utils.transformers import FreqFeature, Binner, Select, DFFeatureUnion
from from sklearn.pipeline import make_pipeline
import pandas as pd

df = pd.DataFrame({
    "sales_category": ['Sale', 'Sale', 'Not Sale'],
    "sales": [1500, 2000, 2250, 7830]
})

freq = make_pipeline(Select('sales_category') ,FreqFeature())
binned = make_pipeline(Select('sales'), Binner(bins=[0, 1000, 2000, 8000]))

union = DFFeatureUnion([freq, binned])
union.fit_transform(df)
```