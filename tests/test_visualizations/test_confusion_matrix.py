import numpy as np
import pytest
from matplotlib import pyplot as plt

from ml_tooling import Model
from ml_tooling.plots import plot_confusion_matrix


class TestConfusionMatrixPlot:
    def test_confusion_matrix_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.confusion_matrix(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_confusion_matrix_plots_have_correct_data(self, classifier: Model):
        ax = classifier.result.plot.confusion_matrix()

        assert "Confusion Matrix - LogisticRegression - Normalized" == ax.title._text
        result = [text._text for text in ax.texts]
        assert pytest.approx(1) == np.round(np.sum([float(x) for x in result]), 1)
        assert {"0.61", "0.32", "0.03", "0.05"} == set(result)
        assert "True Label" == ax.get_ylabel()
        assert "Predicted Label" == ax.get_xlabel()
        plt.close()

    def test_confusion_matrix_plots_have_correct_data_with_threshold(
        self, classifier: Model
    ):
        ax = classifier.result.plot.confusion_matrix(threshold=0.70)

        assert "Confusion Matrix - LogisticRegression - Normalized" == ax.title._text
        result = [text._text for text in ax.texts]
        assert pytest.approx(1) == np.round(np.sum([float(x) for x in result]), 1)
        assert {"0.34", "0.66", "0.00"} == set(result)
        assert "True Label" == ax.get_ylabel()
        assert "Predicted Label" == ax.get_xlabel()
        plt.close()

    def test_confusion_matrix_plots_have_correct_data_when_not_normalized(
        self, classifier: Model
    ):
        ax = classifier.result.plot.confusion_matrix(normalized=False)

        assert "Confusion Matrix - LogisticRegression" == ax.title._text
        result = {text._text for text in ax.texts}
        assert {"23", "12", "1", "2"} == result
        assert "True Label" == ax.get_ylabel()
        assert "Predicted Label" == ax.get_xlabel()
        plt.close()

    def test_confusion_matrix_has_custom_labels(self):
        ax = plot_confusion_matrix(
            y_true=np.array([1, 1, 0, 1]),
            y_pred=np.array([1, 1, 1, 1]),
            labels=["Pos", "Neg"],
        )

        assert "Confusion Matrix - Normalized" == ax.title._text
        assert ["Pos", "Neg"] == [x._text for x in ax.get_xticklabels()]
        assert ["Pos", "Neg"] == [y._text for y in ax.get_yticklabels()]
        plt.close()
