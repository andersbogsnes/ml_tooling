from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from ml_tooling.data import Dataset
from ml_tooling.transformers import DFStandardScaler


class TestMissingDataViz:
    @pytest.fixture()
    def missing_data(self, train_boston_dataset):
        train_boston_dataset.x.iloc[:10, 0] = np.nan
        return train_boston_dataset

    @pytest.fixture()
    def ax(self, missing_data):
        axis = missing_data.plot.missing_data()
        axis.figure.canvas.draw()
        yield axis
        plt.close()

    def test_missing_data_can_pass_pipeline(self, missing_data: Dataset):
        pipeline = Pipeline([("scaler", DFStandardScaler())])
        ax = missing_data.plot.missing_data(feature_pipeline=pipeline)
        assert [text.get_text() for text in ax.texts] == ["2.0%"]
        plt.close()

    def test_target_correlation_uses_pipeline_when_passed(
        self, train_iris_dataset: Dataset
    ):
        mock = MagicMock(spec=Pipeline)
        mock.fit_transform.return_value = train_iris_dataset.x

        train_iris_dataset.plot.target_correlation(feature_pipeline=mock)
        mock.fit_transform.assert_called_once_with(train_iris_dataset.x)
        plt.close()

    def test_missing_data_text_labels_are_correct(self, ax):
        assert [text.get_text() for text in ax.texts] == ["2.0%"]

    def test_ylabel_is_correct(self, ax):
        assert ax.get_ylabel() == "Feature"

    def test_xlabel_is_correct(self, ax):
        assert ax.get_xlabel() == "Percent Missing Data"

    def test_xticklabels_are_correct(self, ax):
        assert [text.get_text() for text in ax.get_xticklabels()] == [
            "0.00%",
            "0.50%",
            "1.00%",
            "1.50%",
            "2.00%",
            "2.50%",
        ]

    def test_missing_data_plots_can_be_given_an_ax(self, missing_data):
        fig, ax = plt.subplots()
        test_ax = missing_data.plot.missing_data(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_can_call_missing_data_with_no_missing_values(self, train_boston_dataset):
        ax = train_boston_dataset.plot.missing_data()
        assert ax.patches == []
        plt.close()
