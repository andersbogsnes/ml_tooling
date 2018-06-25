import pathlib
BASE_PATH = pathlib.Path(__file__).parent

default_config = {
    "VERBOSITY": 0,
    "CLASSIFIER_METRIC": 'accuracy',
    "REGRESSION_METRIC": 'r2',
    "CROSS_VALIDATION": 10,
    "STYLE_SHEET": str(BASE_PATH.joinpath('almbrand.mplstyle')),
    "N_JOBS": -1
}
