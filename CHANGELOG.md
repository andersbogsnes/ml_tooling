v0.5.1
- .train_model will now reset the result attribute to None, in order to 
prevent users from mistakenly assuming the result is from the training 
- Fixed bug in lift_score when using dataframes
- Fixed bug when training model and then scoring model

v0.5.0
- Added Binarize Transformer
- Added ability to use keywords in FuncTransformer
- .predict now returns a dataframe indexed on input
- Updated dependencies

v0.4.0
- Added gridsearch method to BaseClass. Gridsearch your model 
and return a list of results for inspection
- Added ResultGroup - any method that returns a list of results now 
returns a ResultGroup instead.
- Added logging
- Added ability to record runs as yaml

v0.3.2
- Another bugfix release

v0.3.1
- Fixed bug that prevented DFRowFunc from pickling properly

v0.3.0
- Added DFRowFunc Transformer
- Updated FillNA to handle categorical values
- Allow user to choose whether score_model uses cv or not

v0.2.3
- Plot_feature_importance now takes a top_n and bottom_n argument

v0.2.2
- Fix for error in setup wheels

v0.1.3
- Implemented new FillNA Transformer

v0.1.2
- Refactored to use flat structure

v0.1.1
- Renamed project to ml_tooling
    
v0.1.0
- Initial release

    
