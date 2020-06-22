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
from ml_tooling.utils import DatasetError


def load_demo_dataset(dataset_name: str, **kwargs):
    """
    Creates a data class inheriting from ML Tooling dataclas Dataset.
    The demo data is imported from sklearn.datasets

    Parameters
    ----------

    dataset_name: str
        dataset_name defines which dataset should be selected. When 'openml' is selected
        either parameter name or data_id needs to be specified. Possible dataset_name: 'iris',
        'boston', 'diabetes', 'digits', 'linnerud', 'wine', 'breast_cancer', 'openml'


    **kwargs:
        name: str or None
            String identifier of the dataset. Note that OpenML can have multiple datasets
            with the same name.

        version: integer or ‘active’, default=’active’
            Version of the dataset. Can only be provided if also name is given. If
            ‘active’ the oldest version that’s still active is used. Since there may be
            more than one active version of a dataset, and those versions may fundamentally
            be different from one another, setting an exact version is highly recommended.

        data_id: int or None
            OpenML ID of the dataset. The most specific way of retrieving a dataset. If
            data_id is not given, name (and potential version) are used to obtain a dataset.

        data_home: string or None, default None
            Specify another download and cache folder for the data sets. By default all
            scikit-learn data is stored in ‘~/scikit_learn_data’ subfolders.

        target_column: string, list or None, default ‘default-target’
            Specify the column name in the data to use as target. If ‘default-target’, the
            standard target column a stored on the server is used. If None, all columns are
            returned as data and the target is None. If list (of strings), all columns with
            these names are returned as multi-target (Note: not all scikit-learn classifiers
            can handle all types of multi-output combinations)

        cache: boolean, default=True
            Whether to cache downloaded datasets using joblib.

    Returns
    -------
    Dataset
        A data class inheriting from ML Tooling dataclas Dataset

    """
    if kwargs.get("return_X_y"):
        raise DatasetError("return_X_y should be False")

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
        def load_training_data(self):
            return (
                pd.DataFrame(selected_data.data, columns=selected_data.feature_names),
                selected_data.target,
            )

        def load_prediction_data(self, idx):
            x = pd.DataFrame(selected_data.data, columns=selected_data.feature_names)
            return x.loc[idx]

    return DemoData()
