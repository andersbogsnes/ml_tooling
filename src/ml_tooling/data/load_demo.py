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


def load_demo_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Create a :class:`~ml_tooling.data.base_data.Dataset` implementing the demo
    datasets from :ref:`sklearn.datasets <sklearn:datasets>`

    Parameters
    ----------

    dataset_name: str
        Name of the dataset to use. If 'openml' is passed either parameter name or
        data_id needs to be specified.

        One of:
            - iris
            - boston
            - diabetes
            - digits
            - linnerud
            - wine
            - breast_cancer
            - openml

    **kwargs:
        Kwargs are passed on to the scikit-learn dataset function

    Returns
    -------
    Dataset
        An instance of :class:`~ml_tooling.data.Dataset`

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
