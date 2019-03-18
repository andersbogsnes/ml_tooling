.. _plotting:

Plotting
========

When a model is trained, it returns a :class:`Result` object.
That object has number of visualization options depending on the type of model:

Any visualizer listed here also has a functional counterpart in `ml_tooling.plots`.
E.g if you want to use the function for plotting a confusion matrix without using
the ml_tooling BaseClassModel approach, you can instead do:

    from ml_tooling.plots import plot_confusion_matrix


These functional counterparts all mirror sklearn metrics api, taking y_target and y_pred
as arguments:

    from ml_tooling.plots import plot_confusion_matrix
    import numpy as np

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    plot_confusion_matrix(y_true, y_pred)

Classifiers
-----------

- `roc_curve(**kwargs)`:<br />  Visualize a ROC curve for a classification model. Model must implement a `predict_proba` method. Any kwargs are passed onto matplotlib. <br /> <br />
- `confusion_matrix(normalized = True, **kwargs)`:
Visualize a confusion matrix for a classification model. `normalized` determines whether or not to normalize annotated class counts. Any kwargs are passed onto matplotlib.  <br /> <br />
- `feature_importance(samples, values = True,  top_n = None, bottom_n = None, n_jobs_overwrite=None, **kwargs)`:<br /> Calculates each features imporatance with permutation. Importance is measured in drop in model metric. `samples` determines the number of samples to use and must be set. <br /> <br />
If `samples=None` the original data set is used which is not recommended for small data sets. <br /> <br />
If `samples` is a `float` between 0 and 1 a new smaller data set is made from resampling with replacement form the original data set. This is not recommended for small data sets but could be suitable for very large data sets. <br /> <br />
If  `samples` is set to an `int` a new  data set of size `samples` is made from resampling with replacement form the original data. Recommended for small data sets to ensure stable estimates of feature importance.  <br /> <br />
If `top_n` is an `integer`, return `top_n` features and if `top_n` is a `float` between `(0, 1)`, return `top_n` percent features. <br /> <br /> If `bottom_n` is an `integer`, return `bottom_n` features and if `bottom_n` is a `float` between `(0, 1)`, return `bottom_n` percent features. <br /> <br />
Setting `n_jobs_overwrite` to an `int` overwrites the settings of the model settings. <br />

- `lift_curve(**kwargs)`: <br />
Visualize a Lift Curve for a classification model. Model must implement a `predict_proba` method. Any kwargs are passed onto matplotlib.

### Regressors

- `prediction_error(**kwargs)`:<br />
Visualizes prediction error of a regression model. Any kwargs are passed onto matplotlib.
 <br /> <br />
- `residuals(**kwargs)`: <br />
Visualizes residuals of a regression model. Any kwargs are passed onto matplotlib.
 <br /> <br />
- `feature_importance(samples, values = True,  top_n = None, bottom_n = None, n_jobs_overwrite=None, **kwargs)`:<br /> Calculates each features imporatance with permutation. Importance is measured in drop in model metric. `samples` determines the number of samples to use and must be set. <br /> <br />
If `samples=None` the original data set is used which is not recommended for small data sets. <br /> <br />
If `samples` is a `float` between 0 and 1 a new smaller data set is made from resampling with replacement form the original data set. This is not recommended for small data sets but could be suitable for very large data sets. <br /> <br />
If  `samples` is set to an `int` a new  data set of size `samples` is made from resampling with replacement form the original data. Recommended for small data sets to ensure stable estimates of feature importance.  <br /> <br />
If `top_n` is an `integer`, return `top_n` features and if `top_n` is a `float` between `(0, 1)`, return `top_n` percent features. <br /> <br /> If `bottom_n` is an `integer`, return `bottom_n` features and if `bottom_n` is a `float` between `(0, 1)`, return `bottom_n` percent features. <br /> <br />
Setting `n_jobs_overwrite` to an `int` overwrites the settings of the model settings. <br />
