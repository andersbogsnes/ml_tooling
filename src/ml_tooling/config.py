import pathlib
BASE_PATH = pathlib.Path(__file__).parent
MPL_STYLESHEET = str(BASE_PATH.joinpath('almbrand.mplstyle'))

default_config = {
    "VERBOSITY": 0,
    "CLASSIFIER_METRIC": 'accuracy',
    "REGRESSION_METRIC": 'r2',
    "CROSS_VALIDATION": 10,
    "STYLE_SHEET": MPL_STYLESHEET,
    "N_JOBS": -1
}
