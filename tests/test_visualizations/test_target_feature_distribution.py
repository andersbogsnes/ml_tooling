from unittest.mock import MagicMock

import pytest
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from ml_tooling.data import Dataset
from ml_tooling.transformers import Select


class TestTargetFeatureDistribution:
    @pytest.fixture()
    def iris_data(self, iris_df):
        class IrisData(Dataset):
            def load_prediction_data(self):
                return iris_df.drop(columns="target").iloc[:5, 2]

            def load_training_data(self):
                return (
                    iris_df.drop(columns="target").iloc[:5, 2],
                    iris_df.target.iloc[:5],
                )

        return IrisData()

    @pytest.fixture()
    def iris_data_pipeline(self, iris_df):
        class IrisData(Dataset):
            def load_prediction_data(self):
                return iris_df.drop(columns="target").iloc[:5, :]

            def load_training_data(self):
                return (
                    iris_df.drop(columns="target").iloc[:5, :],
                    iris_df.target.iloc[:5],
                )

        return IrisData()

    @pytest.fixture()
    def ax(self, iris_data: Dataset):
        axis = iris_data.plot.target_feature_distribution()
        axis.figure.canvas.draw()
        yield axis
        plt.close()

    def test_target_feature_distribution_can_pass_pipeline(
        self, iris_data_pipeline: Dataset
    ):
        pipeline = Pipeline([("select", Select("petal length (cm)"))])
        ax = iris_data_pipeline.plot.target_feature_distribution(
            feature_pipeline=pipeline
        )
        assert [text.get_text() for text in ax.texts] == ["1.30", "1.40", "1.50"]
        plt.close()

    def test_target_feature_distribution_uses_pipeline_when_passed(
        self, iris_data_pipeline: Dataset
    ):
        mock = MagicMock(spec=Pipeline)
        mock.fit_transform.return_value = iris_data_pipeline.x

        iris_data_pipeline.plot.target_feature_distribution(feature_pipeline=mock)
        mock.fit_transform.assert_called_once_with(iris_data_pipeline.x)
        plt.close()

    def test_target_feature_distribution_works_as_expected(self, ax):
        assert [text.get_text() for text in ax.texts] == ["1.30", "1.40", "1.50"]

        assert ax.title.get_text() == "Target feature distribution"
        assert ax.get_xlabel() == "Target compared to mean"
        assert ax.get_ylabel() == "Feature categories"
        plt.close()

    def test_target_feature_distribution_works_with_different_methods(
        self, iris_data: Dataset
    ):
        ax = iris_data.plot.target_feature_distribution(method="median")

        assert [text.get_text() for text in ax.texts] == ["1.30", "1.40", "1.50"]

        assert ax.title.get_text() == "Target feature distribution"
        assert ax.get_xlabel() == "Target compared to median"
        assert ax.get_ylabel() == "Feature categories"
        plt.close()

    def test_target_feature_distribution_works_with_n_boot(self, iris_data: Dataset):
        ax = iris_data.plot.target_feature_distribution(n_boot=2)

        assert [text.get_text() for text in ax.texts] == ["1.30", "1.40", "1.50"]

        assert ax.title.get_text() == "Target feature distribution"
        assert ax.get_xlabel() == "Target compared to mean"
        assert ax.get_ylabel() == "Feature categories"
        plt.close()

    def test_target_feature_distribution_plots_can_be_given_an_ax(self, iris_data):
        fig, ax = plt.subplots()
        test_ax = iris_data.plot.target_feature_distribution(ax=ax)
        assert ax == test_ax
        plt.close()
