from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset, load_demo_dataset
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import DataType


class TestDemoDatasetModule:
    @pytest.fixture
    def load_dataset_iris(self) -> Dataset:
        return load_demo_dataset("iris")

    @pytest.fixture
    def iris_df(self):
        iris_data = load_iris()
        return (
            pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names),
            iris_data.target,
        )

    def test_repr_is_correct_load(self, load_dataset_iris: Dataset):
        result = str(load_dataset_iris)
        assert result == "<DemoData - Dataset>"

    def test_dataset_return_correct_x_attribute(
        self, load_dataset_iris: Dataset, iris_df: Tuple[pd.DataFrame, DataType]
    ):
        x_expected, y_expected = iris_df
        pd.testing.assert_frame_equal(load_dataset_iris.x, x_expected)

    def test_dataset_return_correct_y_attribute(
        self, load_dataset_iris: Dataset, iris_df: Tuple[pd.DataFrame, DataType]
    ):
        x_expected, y_expected = iris_df
        assert np.array_equal(load_dataset_iris.y, y_expected)

    def test_dataset_from_fetchopenml_works(self):
        dataset = load_demo_dataset("openml", name="miceprotein")
        assert len(dataset.x) == 1080

    def test_dataset_x_from_fetchopenml_with_parameters_works(self):
        dataset = load_demo_dataset(
            "openml", name="blood-transfusion-service-center", target_column="V1"
        )
        features_x = dataset.x
        assert features_x.shape == (748, 4)

    def test_dataset_y_from_fetchopenml_with_two_target_columns_works(self):
        dataset = load_demo_dataset(
            "openml",
            name="blood-transfusion-service-center",
            target_column=["V1", "V2"],
        )
        features_y = dataset.y
        assert features_y.shape == (748, 2)

    def test_load_prediction_data_works_as_expected(self):
        dataset = load_demo_dataset("iris")
        dataset.create_train_test(stratify=True)
        feature_pipeline = Pipeline([("scale", DFStandardScaler())])
        model = Model(LogisticRegression(), feature_pipeline=feature_pipeline)
        model.train_estimator(dataset)
        result = model.make_prediction(dataset, 5)

        expected = pd.DataFrame({"Prediction": [0]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
