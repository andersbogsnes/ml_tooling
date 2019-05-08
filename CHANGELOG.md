# v0.7.0
- Breaking change - BaseClassModel renamed to ModelData.
- Breaking change - model renamed to estimator
- Added Precision-Recall Curve
- Added option to give custom file name to .save_estimator()
- Instantiating with estimator is now optional - set estimator later using .init_estimator

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

    
