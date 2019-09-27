import logging
import random as rand

import numpy as np
import pytest
from ml_tooling.config import DefaultConfig
from ml_tooling import ModelData

logging.disable(logging.CRITICAL)

pytest_plugins = [
    "tests.fixtures.dataframes",
    "tests.fixtures.datasets",
    "tests.fixtures.estimators",
]


@pytest.fixture(autouse=True)
def random():
    rand.seed(42)
    np.random.seed(42)


@pytest.fixture(name="base")
def _base():
    # noinspection PyAbstractClass
    class IrisModel(ModelData):
        @classmethod
        def clean_model(cls):
            cls.config = DefaultConfig()
            cls.config.CROSS_VALIDATION = 2
            cls.config.N_JOBS = 2
            return cls

    return IrisModel.clean_model()


@pytest.fixture
def monkeypatch_git_hash(monkeypatch):
    def mockreturn():
        return "1234"

    monkeypatch.setattr("ml_tooling.utils.get_git_hash", mockreturn)
