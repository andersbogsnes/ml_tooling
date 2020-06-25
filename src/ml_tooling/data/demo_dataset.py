from typing import Tuple

import pandas as pd
from sklearn.datasets import (
    load_boston,
    load_iris,
    load_diabetes,
    load_digits,
    load_linnerud,
    load_wine,
    load_breast_cancer,
    fetch_openml,
)

from ml_tooling.data.base_data import Dataset
from ml_tooling.utils import DataType


def load_demo_dataset(dataset_name: str, **kwargs):
    """
    The function calls either the load_ or fetch_ function from
    :class:`sklearn.datasets`. Then a data class is created which is inheriting from
    ML Tooling dataclas Dataset.
    The options in kwargs can be read in :class:`sklearn.datasets`

    Parameters
    ----------

    dataset_name: str
        dataset_name defines which dataset should be selected. When 'openml' is selected
        either parameter name or data_id needs to be specified. Possible dataset_name: 'iris',
        'boston', 'diabetes', 'digits', 'linnerud', 'wine', 'breast_cancer', 'openml'

    **kwargs:

    Returns
    -------
    Dataset
        A data class inheriting from ML Tooling dataclas Dataset

    """

    dataset_mapping = {
        "iris": load_iris,
        "boston": load_boston,
        "diabetes": load_diabetes,
        "digits": load_digits,
        "linnerud": load_linnerud,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "openml": fetch_openml,
    }

    selected_data = dataset_mapping[dataset_name](**kwargs)

    class DemoData(Dataset):
        def load_training_data(self) -> Tuple[pd.DataFrame, DataType]:
            return (
                pd.DataFrame(selected_data.data, columns=selected_data.feature_names),
                selected_data.target,
            )

        def load_prediction_data(self, idx) -> pd.DataFrame:
            x = pd.DataFrame(selected_data.data, columns=selected_data.feature_names)
            return x.loc[[idx]]

    return DemoData()
