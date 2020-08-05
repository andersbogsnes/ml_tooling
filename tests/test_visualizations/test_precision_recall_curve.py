import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC

from ml_tooling import Model
from ml_tooling.plots import plot_pr_curve
from ml_tooling.utils import VizError


class TestPRCurve:
    def test_prediction_error_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.pr_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_pr_curve_have_correct_data(self, classifier: Model):
        ax = classifier.result.plot.pr_curve()
        x, y = classifier.result.plot._data.test_x, classifier.result.plot._data.test_y
        y_proba = classifier.estimator.predict_proba(x)[:, 1]

        precision, recall, _ = precision_recall_curve(y, y_proba)

        assert "Precision-Recall - LogisticRegression" == ax.title.get_text()
        assert "Precision" == ax.get_ylabel()
        assert "Recall" == ax.get_xlabel()
        assert np.all(recall == ax.lines[0].get_xdata())
        assert np.all(precision == ax.lines[0].get_ydata())
        plt.close()

    def test_pr_curve_fails_correctly_without_predict_proba(self, train_iris_dataset):
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(train_iris_dataset)
        with pytest.raises(VizError):
            result.plot.pr_curve()
        plt.close()

    def test_pr_curve_can_use_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        x, y = classifier.result.plot._data.test_x, classifier.result.plot._data.test_y
        y_proba = classifier.estimator.predict_proba(x)[:, 1]

        assert ax is plot_pr_curve(y, y_proba, ax=ax)
        plt.close()
