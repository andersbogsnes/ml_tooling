import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.svm import SVC

from ml_tooling import Model
from ml_tooling.utils import VizError


class TestRocCurve:
    def test_roc_curve_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.roc_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_roc_curve_have_correct_data(self, classifier: Model):
        ax = classifier.result.plot.roc_curve()
        x, y = classifier.result.plot._data.test_x, classifier.result.plot._data.test_y
        y_proba = classifier.estimator.predict_proba(x)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)

        assert "ROC AUC - LogisticRegression" == ax.title._text
        assert "True Positive Rate" == ax.get_ylabel()
        assert "False Positive Rate" == ax.get_xlabel()
        assert np.all(fpr == ax.lines[0].get_xdata())
        assert np.all(tpr == ax.lines[0].get_ydata())
        plt.close()

    def test_roc_curve_fails_correctly_without_predict_proba(self, train_iris_dataset):
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(train_iris_dataset)
        with pytest.raises(VizError):
            result.plot.roc_curve()
        plt.close()