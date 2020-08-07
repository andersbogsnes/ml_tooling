from unittest.mock import MagicMock

from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from ml_tooling.data import Dataset
from ml_tooling.transformers import DFStandardScaler


class TestTargetCorrelation:
    def test_target_correlation_can_pass_pipeline(self, train_iris_dataset: Dataset):
        pipeline = Pipeline([("scaler", DFStandardScaler())])
        ax = train_iris_dataset.plot.target_correlation(feature_pipeline=pipeline)
        assert [text.get_text() for text in ax.texts] == [
            "0.01",
            "0.02",
            "0.12",
            "-0.48",
        ]
        plt.close()

    def test_target_correlation_uses_pipeline_when_passed(
        self, train_iris_dataset: Dataset
    ):
        mock = MagicMock(spec=Pipeline)
        mock.fit_transform.return_value = train_iris_dataset.x

        train_iris_dataset.plot.target_correlation(feature_pipeline=mock)
        mock.fit_transform.assert_called_once_with(train_iris_dataset.x)
        plt.close()

    def test_target_correlation_works_as_expected(self, train_iris_dataset):
        ax = train_iris_dataset.plot.target_correlation()

        assert [text.get_text() for text in ax.texts] == [
            "0.01",
            "0.02",
            "0.12",
            "-0.48",
        ]

        assert ax.title.get_text() == "Feature-Target Correlation"
        assert ax.get_xlabel() == "Spearman Correlation"
        assert ax.get_ylabel() == "Feature Labels"
        plt.close()

    def test_target_correlation_works_with_different_methods(self, train_iris_dataset):
        ax = train_iris_dataset.plot.target_correlation(method="pearson")

        assert [text.get_text() for text in ax.texts] == [
            "0.08",
            "0.12",
            "0.20",
            "-0.47",
        ]
        assert ax.title.get_text() == "Feature-Target Correlation"
        assert ax.get_xlabel() == "Pearson Correlation"
        assert ax.get_ylabel() == "Feature Labels"
        plt.close()

    def test_target_correlation_works_with_top_n(self, train_iris_dataset):
        ax = train_iris_dataset.plot.target_correlation(top_n=2)
        assert [text.get_text() for text in ax.texts] == ["0.12", "-0.48"]
        assert ax.title.get_text() == "Feature-Target Correlation - Top 2"
        assert ax.get_xlabel() == "Spearman Correlation"
        assert ax.get_ylabel() == "Feature Labels"
        plt.close()

    def test_target_correlation_works_with_bottom_n(self, train_iris_dataset):
        ax = train_iris_dataset.plot.target_correlation(bottom_n=2)
        assert [text.get_text() for text in ax.texts] == ["0.01", "0.02"]
        assert ax.title.get_text() == "Feature-Target Correlation - Bottom 2"
        assert ax.get_xlabel() == "Spearman Correlation"
        assert ax.get_ylabel() == "Feature Labels"
        plt.close()

    def test_target_correlation_works_with_bottom_n_and_top_n(self, train_iris_dataset):
        ax = train_iris_dataset.plot.target_correlation(bottom_n=1, top_n=1)
        assert [text.get_text() for text in ax.texts] == ["0.01", "-0.48"]
        assert ax.title.get_text() == "Feature-Target Correlation - Top 1 - Bottom 1"
        assert ax.get_xlabel() == "Spearman Correlation"
        assert ax.get_ylabel() == "Feature Labels"
        plt.close()

    def test_target_correlation_plots_can_be_given_an_ax(self, train_iris_dataset):
        fig, ax = plt.subplots()
        test_ax = train_iris_dataset.plot.target_correlation(ax=ax)
        assert ax == test_ax
        plt.close()
