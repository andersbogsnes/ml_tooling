import pathlib

BASE_PATH = pathlib.Path(__file__).parent
MPL_STYLESHEET = str(BASE_PATH.joinpath('almbrand.mplstyle'))


class DefaultConfig:
    """
    Configuration for a given BaseClass. Configs propagate through each instance
    """
    def __init__(self):
        self.VERBOSITY = 0
        self.CLASSIFIER_METRIC = 'accuracy'
        self.REGRESSION_METRIC = 'r2'
        self.CROSS_VALIDATION = 10
        self.STYLE_SHEET = MPL_STYLESHEET
        self.N_JOBS = -1
        self.TEST_SIZE = 0.25
        self.RANDOM_STATE = 42

    def __repr__(self):
        attrs = '\n'.join([f"{attr}: {value}"
                           for attr, value in self.__dict__.items()
                           if '__' not in attr])
        return f'<Config: \n{attrs}\n>'
