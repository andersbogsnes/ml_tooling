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


def log_model(metric_scores: dict,
              model_name: str,
              model_params: dict,
              run_dir: Union[pathlib.Path, str],
              model_path=None,
              overwrite=False,
              ):
    from ml_tooling import __version__ as ml_tools_version
    from sklearn import __version__ as sklearn_version
    from pandas import __version__ as pandas_version

    versions = {"ml_tooling": ml_tools_version,
                "sklearn": sklearn_version,
                "pandas": pandas_version}

    data = {"time_created": datetime.now(),
            "estimator_name": model_name,
            "versions": versions,
            "params": model_params,
            "git_hash": get_git_hash(),
            "metrics": {k: float(v) for k, v in metric_scores.items()}
            }

    if model_path:
        data['model_path'] = str(model_path)

    now = datetime.now()
    metrics = '_'.join([f'{k}_{v:.3f}' for k, v in metric_scores.items()])
    run_dir = pathlib.Path(run_dir)

    run_dir = _make_run_dir(run_dir.joinpath(now.strftime('%Y%m%d')))
    output_file = f'{model_name}_{metrics}_{now.strftime("%H%M")}.yaml'
    output_path = run_dir.joinpath(output_file)

    if output_path.exists() and not overwrite:
        output_file = f'{model_name}_{metrics}_{now.strftime("%H%M%S")}.yaml'
        output_path = output_path.with_name(output_file)

    with output_path.open(mode='w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    return output_path
