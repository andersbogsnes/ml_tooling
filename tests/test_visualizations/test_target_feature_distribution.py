from unittest.mock import MagicMock
import pytest

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml

from ml_tooling.data import Dataset


class TestTargetFeatureDistribution:
    @pytest.fixture()
    def australian_dataset(self):
        australian_data = fetch_openml("Australian", as_frame=True)
        australian_data_df = pd.DataFrame(
            data=australian_data.data, columns=australian_data.feature_names
        )
        australian_data_df_target = australian_data.target.astype("int")

        class AustralianData(Dataset):
            def load_training_data(self):
                return australian_data_df, australian_data_df_target

            def load_prediction_data(self):
                return australian_data_df

        return AustralianData()

    def test_target_correlation_uses_pipeline_when_passed(
        self, train_iris_dataset: Dataset
    ):
        mock = MagicMock(spec=Pipeline)
        mock.fit_transform.return_value = train_iris_dataset.x["petal length (cm)"]

        train_iris_dataset.plot.target_feature_distribution(
            feature_name="petal length (cm)", feature_pipeline=mock
        )
        mock.fit_transform.assert_called_once_with(
            train_iris_dataset.x["petal length (cm)"]
        )
        plt.close()

    def test_target_feature_distribution_works_as_expected(
        self, australian_dataset: Dataset
    ):
        ax = australian_dataset.plot.target_feature_distribution(feature_name="A4")

        assert [text.get_text() for text in ax.texts] == ["0.28", "0.50", "1.00"]

        assert ax.title.get_text() == "Target feature distribution"
        assert ax.get_xlabel() == "mean target"
        assert ax.get_ylabel() == "Feature categories"
        plt.close()

    def test_target_feature_distribution_works_with_different_methods(
        self, australian_dataset: Dataset
    ):
        ax = australian_dataset.plot.target_feature_distribution(
            feature_name="A4", method="median"
        )

        assert [text.get_text() for text in ax.texts] == ["0.00", "0.00", "1.00"]

        assert ax.title.get_text() == "Target feature distribution"
        assert ax.get_xlabel() == "median target"
        assert ax.get_ylabel() == "Feature categories"
        plt.close()

    def test_target_feature_distribution_plots_can_be_given_an_ax(
        self, australian_dataset: Dataset
    ):
        fig, ax = plt.subplots()

        test_ax = australian_dataset.plot.target_feature_distribution(
            feature_name="A4", ax=ax
        )
        assert ax == test_ax
        plt.close()
