import pathlib
from datetime import datetime
from typing import Union

import yaml
from ml_tooling.utils import get_git_hash


def _make_run_dir(run_dir: str) -> pathlib.Path:
    path = pathlib.Path(run_dir)

    if path.is_file():
        raise IOError(f"{run_dir} is a file - pass a directory")

    if not path.exists():
        path.mkdir(parents=True)

    return path


def create_log(
    name: str,
    metric_scores: dict,
    serialized_estimator: dict,
    saved_estimator_path: pathlib.Path,
) -> dict:
    from ml_tooling import __version__ as ml_tools_version
    from sklearn import __version__ as sklearn_version
    from pandas import __version__ as pandas_version

    versions = {
        "ml_tooling": ml_tools_version,
        "sklearn": sklearn_version,
        "pandas": pandas_version,
    }

    data = {
        "model_name": name,
        "time_created": datetime.now(),
        "versions": versions,
        "git_hash": get_git_hash(),
        "metrics": {k: float(v) for k, v in metric_scores.items()},
        "estimator": serialized_estimator,
        "estimator_path": str(saved_estimator_path) if saved_estimator_path else None,
    }
    return data


def save_log(log: dict, save_dir: pathlib.Path) -> pathlib.Path:
    now = datetime.now()
    save_dir = pathlib.Path(save_dir)
    save_dir = _make_run_dir(save_dir / now.strftime("%Y%m%d"))

    iteration = 0
    output_file = f'{log["model_name"]}_{now.strftime("%H%M%S")}_{iteration}.yaml'
    output_path = save_dir.joinpath(output_file)

    while output_path.exists():
        output_file = f'{log["model_name"]}_{now.strftime("%H%M%S")}_{iteration}.yaml'
        output_path = save_dir.joinpath(output_file)
        iteration += 1

    with output_path.open(mode="w") as f:
        yaml.safe_dump(log, f, default_flow_style=False, allow_unicode=True)

    return output_path


def log_results(
    name: str,
    metric_scores: dict,
    serialized_estimator: dict,
    save_dir: Union[pathlib.Path, str],
    saved_estimator_path: pathlib.Path = None,
):
    """
    Logs information about the result of a model in a .yaml file
    .. todo::

        Show parameters being logged

    .. todo::

        Fill out docstring

    Parameters
    ----------
    name: str
        Name used to identify estimator
    metric_scores: dict
        Dictionary of metric name -> score
    serialized_estimator: dict
        Dictionary containing the serialized version of the estimator
    save_dir: pathlib.Path
        Where to save the logging file
    saved_estimator_path: pathlib.Path
        Where the estimator pickle file has been saved

    Returns
    -------
    pathlib.Path
        Path where output has been saved
    """
    log = create_log(name, metric_scores, serialized_estimator, saved_estimator_path)
    output_path = save_log(log, save_dir)
    return output_path
