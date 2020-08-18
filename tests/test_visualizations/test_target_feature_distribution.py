from unittest.mock import MagicMock

from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from ml_tooling.data import Dataset


class TestTargetFeatureDistribution:
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
        assert ax.get_xlabel() == "Target compared to mean"
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
        assert ax.get_xlabel() == "Target compared to median"
        assert ax.get_ylabel() == "Feature categories"
        plt.close()

    def test_target_feature_distribution_works_with_n_boot(
        self, australian_dataset: Dataset
    ):
        ax = australian_dataset.plot.target_feature_distribution(
            feature_name="A4", n_boot=2
        )

        assert [text.get_text() for text in ax.texts] == ["0.28", "0.50", "1.00"]

        assert ax.title.get_text() == "Target feature distribution"
        assert ax.get_xlabel() == "Target compared to mean"
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
