import logging
import pathlib
import random as rand

import numpy as np
import pytest
from ml_tooling.config import DefaultConfig
from ml_tooling import Model

logging.disable(logging.CRITICAL)

pytest_plugins = ["fixtures.dataframes", "fixtures.datasets", "fixtures.estimators"]


@pytest.fixture(autouse=True)
def random():
    rand.seed(42)
    np.random.seed(42)


@pytest.fixture(name="base")
def _base():
    class IrisModel(Model):
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


@pytest.fixture(params=["setup.py", "pyproject.toml"])
def temp_project_structure(tmp_path: pathlib.Path, request) -> pathlib.Path:
    # Write a setup.py file
    tmp_path.joinpath(request.param).write_text("my_setup\nfile")

    # Setup a src project
    project = tmp_path.joinpath("src").joinpath("my_test_project")
    project.mkdir(parents=True)

    # Make sure there's an __init__ file there
    project.joinpath("__init__.py").write_text("__version__ == 0.1.0")
    return tmp_path
