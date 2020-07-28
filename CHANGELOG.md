# v0.11.1
- Permutation importance and Feature importance are now two different plotting methods.
- `Model.test_estimators` now takes a `feature_pipeline` argument
# v0.11.0
- Added `load_demo_dataset` function
- If the dataset has no train set `score_estimator` will now run `create_train_test` with default configurations
- `Model.make_prediction` now takes a threshold argument when making a binary classification
- All ML-tooling logging messages now go to stdout instead of stderr
- Can pass a feature pipeline to `Model` which will then automatically generate a
 combined feature_pipeline + estimator Pipeline
- Can pass a feature pipeline to `Dataset.plot` methods, to apply preprocessing
before visualization
- New config implementation. If you need to reset the configuration, you should use `Model.config.reset_config()`

# v0.10.3
- Fixed typehints in Dataset
- Dataset.create_train_test now takes a boolean `stratify` parameter.
- Added default local filestorage when using `save_estimator`
- The dataframe returned by `.make_prediction`now labels the columns in a more
human friendly manner
- Dataset now verifies that `load_training_data` and `load_prediction_data` do not return empty
- Added a missing data visualization to `Dataset.plot`
- FillNA now accepts a `is_nan`flag which adds a flag indicating that a value was missing
- `Model.make_prediction` now accepts a `use_cache`flag to score everything in cached `.x`
- Added a new Transformer:  `RareFeatureEncoder`

# v0.10.2
- Fixed type inferences from data to sql in _load_data
- Added idx arg to load_prediction_data abstract method in SQLDataset
- Added caching of loaded data in SQLDataset

# v0.10.1
- Added `.copy_to` functionality to SQLDataset and FileDataset,
allowing copying between datasets

# v0.10.0
- Bug fix for calculating feature importance when passing large amounts of data
- Bug fix when using default metric in `test_estimators`
- Bug fix when gridsearching, only applying last change
- Add nicer error message when passing incorrect dtypes to FillNA
- Storage .save method now only takes filename as parameter
- Handles storage loading of paths outputted from the Storage .get_list method
- Handles case when Dataset does not have a `y` value
- Added `plot_learning_curve` and corresponding `result.plot.learning_curve`
- Added `plot_validation_curve` and corresponding `result.plot.validation_curve`
- Replaced `permutation_importance` with scikit-learn's implementation
- Added `target_correlation` plots to Dataset.plot


# v0.9.2
  - Bug fix for logging when feature unions (DFFeatureUnion) had tuples

# v0.9.1
- Hot fix python version to 3.7

# v0.9.0
- Breaking change - Model methods load_estimator and save_estimator now takes a Storage class that defines how and where to store estimators.
- Added the ability to declare that a saved model should be a production estimator.
- Added corresponding `.load_production_estimator` to `Model`

# v0.8.1
- Removed gitpython as a dependency

# v0.8.0
- Replaced custom feature permutation importance with sklearns implementation from v0.22
- Breaking change - Dataset is now a separate object that has to be instantiated outside Modeldata
- Breaking change - ModelData is now renamed to Model
- Added new properties `is_estimator` and `is_regressor` which checks what type of estimator is used

# v0.7.1
- Joblib is now a dependency, instead of being vendored with scikit-learn
- Updated requirements
# v0.7.0
- Breaking change - BaseClassModel renamed to ModelData.
- Breaking change - model renamed to estimator
- Added Precision-Recall Curve
- Added option to give custom file name to .save_estimator()
- Instantiating with estimator is now optional - set estimator later using .init_estimator
- We have a logo! Added ML Tooling logo to docs

# v0.6.2
## Bugfixes
- Now issues a warning when git is not installed.

# v0.6.1

## Bugfixes
- Data for a class is changed from instance variable to class variable
- Grid search only copies data to workers once and reuses them across grid and folds.
- The Data Class now takes a random seed which it will receive from the BaseClass
- Disabled mem-maping in feature importance
- Added license file to package
- Updated requirements

# v0.6.0

## Features
- Feature importances changed to use permutation instead of built-in for better estimates.

## Bugfixes
- .train_estimator will now reset the result attribute to None, in order to
prevent users from mistakenly assuming the result is from the training
- Fixed bug in lift_score when using dataframes
- Fixed bug when training model and then scoring model
- Fixed bug where users could not save models if no result had been created, as would
happen if the user only called .train_estimator before saving.
- Default_metric is now the same metric as the one specified for the model in .config
- Each class inheriting from ModelData has an individual config
- Changed get_scorer_func to wrap sklearn's get_scorer
- Fixed bug when gridsearching twice

# v0.5.0
- Added Binarize Transformer
- Added ability to use keywords in FuncTransformer
- .predict now returns a dataframe indexed on input
- Updated dependencies

# v0.4.0
- Added gridsearch method to BaseClass. Gridsearch your model
and return a list of results for inspection
- Added ResultGroup - any method that returns a list of results now
returns a ResultGroup instead.
- Added logging
- Added ability to record runs as yaml

# v0.3.2
- Another bugfix release

# v0.3.1
- Fixed bug that prevented DFRowFunc from pickling properly

# v0.3.0
- Added DFRowFunc Transformer
- Updated FillNA to handle categorical values
- Allow user to choose whether score_model uses cv or not

# v0.2.3
- Plot_feature_importance now takes a top_n and bottom_n argument

# v0.2.2
- Fix for error in setup wheels

# v0.1.3
- Implemented new FillNA Transformer

# v0.1.2
- Refactored to use flat structure

# v0.1.1
- Renamed project to ml_tooling

# v0.1.0
- Initial release
