import importlib
import pathlib
import subprocess
from subprocess import CalledProcessError
from typing import Union, Tuple, List, Any
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import warnings

DataType = Union[pd.Series, np.ndarray]
Estimator = Union[BaseEstimator, Pipeline]
Pathlike = Union[str, pathlib.Path]

logger = logging.getLogger("ml_tooling")


class MLToolingError(Exception):
    """Error which occurs when using ML Tooling"""


class TransformerError(MLToolingError):
    """Error which occurs during a transform"""


class DatasetError(MLToolingError):
    """Error which occurs when using a DataSet"""


class VizError(MLToolingError):
    """Error which occurs when using a Visualization"""


def read_yaml(filepath: Pathlike) -> dict:
    """
    Loads a yaml file safely, returning a dictionary
    Parameters
    ----------
    filepath: Pathlike
        Location of yaml file

    Returns
    -------
    dict
    """
    log_file = pathlib.Path(filepath)
    with log_file.open("r") as f:
        return yaml.safe_load(f)


def make_pipeline_from_definition(definitions: List[dict]) -> Estimator:
    """
    Goes through each step of a list of estimator definitions and deserialized them.

    Parameters
    ----------
    definitions: List of dicts
        List of serialized estimators to deserialize

    Returns
    -------
    Estimator
    Deserialized estimator
    """
    steps = [_import_pipeline_step(definition) for definition in definitions]
    if len(steps) == 1:
        return steps[0]
    return Pipeline(steps)


def get_git_hash() -> str:
    """
    Returns the git hash of HEAD

    Returns
    -------
    str
        git hash value of HEAD
    """
    try:
        label = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("ascii")
        )
    except (OSError, FileNotFoundError):
        warnings.warn("Error using git - is `git` installed?")
        label = ""
    except CalledProcessError:
        warnings.warn("Error using git - skipping git hash. Did you call `git init`?")
        label = ""
    return label


def _get_estimator_name(clf: Estimator) -> str:
    """
    Returns estimator name based on class name. If passed classifier is a :class:
    `~sklearn.pipeline.Pipeline`, assume last step is the estimator and return that classes name

    Parameters
    ----------
    clf: BaseEstimator, Pipeline

    Returns
    -------
    str
        Name of estimator
    """
    class_name = clf.__class__.__name__

    if class_name == "Pipeline":
        return clf.steps[-1][1].__class__.__name__

    return class_name


def listify(collection) -> list:
    """
    Takes a given collection and returns a list of the elements, handling strings correctly

    Parameters
    ----------
    collection: tuple, set, str
        Any type of collection or string

    Returns
    -------
    list
    """
    if isinstance(collection, str):
        collection = [collection]

    if isinstance(collection, (tuple, set)):
        collection = list(collection)

    return collection


def _validate_estimator(estimator: Estimator):
    """
    Ensures that estimator is a valid estimator - either a :class:`~sklearn.base.BaseEstimator`
    or a :class:`~sklearn.pipeline.Pipeline` with a :class:`~sklearn.base.BaseEstimator`
    as the final step

    Parameters
    ----------
    estimator: passed estimator to validate

    Returns
    -------
    :class:`~sklearn.base.BaseEstimator`

    Raises
    ------
    MLToolingError
        Raises on invalid input
    """

    if hasattr(estimator, "_estimator_type"):
        return estimator

    if isinstance(estimator, Pipeline):
        raise MLToolingError(
            "You passed a Pipeline without an estimator as the last step"
        )

    raise MLToolingError(f"Expected a Pipeline or Estimator - got {type(estimator)}")


def is_pipeline(estimator: Estimator):
    if type(estimator).__name__ == "Pipeline":
        return True
    return False


def _import_pipeline_step(
    definition: dict,
) -> Union[Tuple[str, BaseEstimator], BaseEstimator]:
    """
    Hydrates a class based on a dictionary definition, importing the module
    and instantiating the class from the classname, setting the parameters of the class as found
    in the input dictionary

    Parameters
    ----------
    definition: dict
        Dictionary definition including module name, classname and params. If pipeline is defined,
        returns

    Returns
    -------
    BaseEstimator or (str, BaseEstimator)
        Instantiated BaseEstimator and optionally the name of the step

    """
    module = importlib.import_module(definition["module"])

    if definition["classname"] == "DFFeatureUnion":
        transformer_list = [
            _import_pipeline_step(transformer) for transformer in definition["params"]
        ]
        class_ = getattr(module, definition["classname"])(transformer_list)

    elif definition["classname"] == "Pipeline":
        steps = [
            _import_pipeline_step(transformer) for transformer in definition["params"]
        ]
        class_ = getattr(module, definition["classname"])(steps=steps)
    else:
        class_ = getattr(module, definition["classname"])()
        class_ = class_.set_params(**definition["params"])

    if "name" in definition:
        return definition["name"], class_
    return class_


def _extract_params(transformer: Tuple[str, Any]) -> Union[List[dict], dict]:
    """
    Helper function to extract parameters from a given transformer, depending on whether
    it's a Pipeline, BaseEstimator or FeatureUnion
    Parameters
    ----------
    transformer:
        (`name`, `transformer`) tuple where `transformer` has the
        attribute `transformer_list`, `steps` or
        has the method `get_params`.

    """
    if hasattr(transformer[1], "transformer_list"):
        extract = [serialize_pipeline(s)[0] for s in transformer[1].transformer_list]
    elif hasattr(transformer[1], "steps"):
        extract = [serialize_pipeline(s)[0] for s in transformer[1].steps]
    else:
        extract = transformer[1].get_params()
    return extract


def serialize_pipeline(pipe: Pipeline) -> List[dict]:
    """
    Serialize a pipeline to a dictionary.
    If a FeatureUnion is present, recursively serialize its transformer list

    Parameters
    ----------
    pipe: Pipeline
        Pipeline to serialize

    Returns
    -------
    List of dicts
    """

    if isinstance(pipe, tuple):
        transformer_list = [pipe]
    else:
        transformer_list = pipe.steps

    return_list = []

    for transformer in transformer_list:
        name = transformer[0]
        module = transformer[1].__class__.__module__
        classname = transformer[1].__class__.__name__
        params = _extract_params(transformer)

        return_list.append(
            {"name": name, "module": module, "classname": classname, "params": params}
        )
    return return_list


def make_dir(path: pathlib.Path) -> pathlib.Path:
    """
    Checks that path is a directory and then creates that directory if it doesn't exist
    Parameters
    ----------
    path: pathlib.Path
        Directory to check

    Returns
    -------
    pathlib.Path
        Path to directory
    """

    if path.is_file():
        raise IOError(f"{path} is a file - must pass a directory")

    if not path.exists():
        path.mkdir(parents=True)

    return path


def _find_setup_file(path: pathlib.Path, level: int, max_level: int) -> pathlib.Path:
    """
    Finds project setup.py from path by recursively searching up the file hierarchy

    Parameters
    ----------
    path
    level
    max_level

    Raises
    ------
    MLToolingError
        Raised when we have recursed past max_level and still haven't found a setup.py file

    Returns
    -------
    pathlib.Path
        Path to folder that has a setup.py file in it
    """
    if level > max_level:
        raise MLToolingError("Exceeded max_level. Does your project have a setup.py?")
    if path.joinpath("setup.py").exists():
        return path

    return _find_setup_file(path.parent, level + 1, max_level)


def _find_src_dir(path: pathlib.Path = None, max_level: int = 2) -> pathlib.Path:
    """
    Returns the path to the src dir - recursively searches up the file hierarchy
    until it finds a src dir and then traverses that to find the module
    as defined by having an __init__.py file in the root

    Parameters
    ----------

    path: pathlib.Path
        Path to begin the search - defaults to current working directory

    max_level: int
        How many levels of the file hierarchy to search.
        Too high risks finding unrelated projects,
        too low risks not navigating high enough to reach top src folder.

    Returns
    -------
    pathlib.Path
        Path to project folder
    """
    current_path = pathlib.Path.cwd() if path is None else path
    setup_path = _find_setup_file(current_path, 0, max_level)

    output_folder = setup_path / "src"
    if not output_folder.exists():
        raise MLToolingError("Project must have a src folder!")

    # Make sure there's an __init__ file in the project
    for child in output_folder.glob("*"):
        if child.joinpath("__init__.py").exists():
            return child

    raise MLToolingError(
        f"No modules found in {output_folder}! Is there an __init__.py file in your module?"
    )


def _classify(x: pd.DataFrame, estimator: BaseEstimator, threshold: float = 0.5):
    """
    Make a binary classification of prediction probabilities with the given threshold

    Parameters
    ----------
    x
    estimator
    threshold

    Returns
    -------
    np.array
        Array of predictions
    """
    y_prob = estimator.predict_proba(x)
    y_pred = np.where(
        (y_prob > threshold) & (y_prob == y_prob.max(axis=1, keepdims=True)), 1, 0
    )
    return estimator.classes_[np.argmax(y_pred, axis=1)]
