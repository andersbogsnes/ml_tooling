"""
Test file for visualisations
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.plots import plot_lift_curve, plot_confusion_matrix, plot_pr_curve
from ml_tooling.plots.utils import VizError
from ml_tooling.result.viz import RegressionVisualize, ClassificationVisualize
from ml_tooling.transformers import ToCategorical


class TestVisualize:
    def test_result_regression_gets_correct_visualizers(self, regression: Model):
        result = regression.result
        assert isinstance(result.plot, RegressionVisualize)

    def test_result_classification_gets_correct_visualizers(self, classifier: Model):
        result = classifier.result
        assert isinstance(result.plot, ClassificationVisualize)

    @pytest.mark.parametrize(
        "attr", ["residuals", "prediction_error", "feature_importance"]
    )
    def test_regression_visualize_has_all_plots(self, attr: str, regression: Model):
        result = regression.result.plot
        plotter = getattr(result, attr)()
        assert isinstance(plotter, Axes)
        plt.close()

    @pytest.mark.parametrize(
        "attr",
        [
            "confusion_matrix",
            "roc_curve",
            "lift_curve",
            "feature_importance",
            "pr_curve",
        ],
    )
    def test_classifier_visualize_has_all_plots(self, attr: str, classifier: Model):
        result = classifier.result.plot
        plotter = getattr(result, attr)()
        assert isinstance(plotter, Axes)
        plt.close()


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
        assert {"0.63", "0.18", "0.11", "0.08"} == set(result)
        assert "True Label" == ax.get_ylabel()
        assert "Predicted Label" == ax.get_xlabel()
        plt.close()

    def test_confusion_matrix_plots_have_correct_data_when_not_normalized(
        self, classifier: Model
    ):
        ax = classifier.result.plot.confusion_matrix(normalized=False)

        assert "Confusion Matrix - LogisticRegression" == ax.title._text
        result = {text._text for text in ax.texts}
        assert {"24", "7", "4", "3"} == result
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
        assert ["", "Pos", "Neg", ""] == [x._text for x in ax.get_xticklabels()]
        assert ["", "Pos", "Neg", ""] == [y._text for y in ax.get_yticklabels()]
        plt.close()


class TestFeatureImportancePlot:
    def test_feature_importance_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.feature_importance(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_feature_importance_plots_have_correct_data(self, classifier: Model):
        ax = classifier.result.plot.feature_importance()

        expected = {"0.04", "0.08", "-0.03", "0.02"}
        assert {text._text for text in ax.texts} == expected
        assert "Feature Importance - LogisticRegression" == ax.title._text
        assert "Features" == ax.get_ylabel()
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_have_no_labels_if_value_is_false(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(values=False)
        assert len(ax.texts) == 0
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        assert ax.title._text == "Feature Importance - LogisticRegression"
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_top_n_is_set(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=2)
        assert 2 == len(ax.texts)
        assert {text._text for text in ax.texts} == {"0.04", "0.08"}
        assert ax.title._text == "Feature Importance - LogisticRegression - Top 2"
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_top_n_is_percent(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=0.2)
        assert len(ax.texts) == 1
        assert {text._text for text in ax.texts} == {"0.08"}
        assert ax.title._text == "Feature Importance - LogisticRegression - Top 20%"
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_bottom_n_is_int(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(bottom_n=2)
        assert len(ax.texts) == 2
        assert {text._text for text in ax.texts} == {"0.02", "-0.03"}
        assert ax.title._text == "Feature Importance - LogisticRegression - Bottom 2"
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_bottom_n_is_percent(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(bottom_n=0.2)
        assert len(ax.texts) == 1
        assert {text._text for text in ax.texts} == {"0.02"}
        assert ax.title._text == "Feature Importance - LogisticRegression - Bottom 20%"
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_correct_if_top_n_is_int_and_bottom_n_is_int(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=1, bottom_n=1)
        assert len(ax.texts) == 2
        assert {text._text for text in ax.texts} == {"0.08", "0.02"}
        assert (
            ax.title._text
            == "Feature Importance - LogisticRegression - Top 1 - Bottom 1"
        )
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_correct_when_top_n_is_int_and_bottom_n_is_percent(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=1, bottom_n=0.2)
        assert 2 == len(ax.texts)
        assert {text._text for text in ax.texts} == {"0.08", "0.02"}
        assert (
            ax.title._text
            == "Feature Importance - LogisticRegression - Top 1 - Bottom 20%"
        )
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_correct_when_top_n_is_percent_and_bottom_n_is_int(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=0.2, bottom_n=1)
        assert len(ax.texts) == 2
        assert {text._text for text in ax.texts} == {"0.02", "0.08"}
        assert (
            ax.title._text
            == "Feature Importance - LogisticRegression - Top 20% - Bottom 1"
        )
        assert ax.get_ylabel() == "Features"
        assert ax.get_xlabel() == "Permuted Feature Importance Relative to Baseline"
        plt.close()

    def test_feature_importance_plots_correctly_in_pipeline(
        self, categorical: Model, test_dataset: Dataset
    ):
        pipe = Pipeline(
            [
                ("tocategory", ToCategorical()),
                ("clf", RandomForestClassifier(n_estimators=10)),
            ]
        )

        model = Model(pipe)
        result = model.score_estimator(test_dataset)
        ax = result.plot.feature_importance()

        assert "Feature Importance - RandomForestClassifier" == ax.title._text
        assert 4 == len(ax.get_yticklabels())
        plt.close()


class TestLiftCurvePlot:
    def test_lift_curve_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.lift_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_lift_curve_have_correct_data(self, classifier: Model):
        ax = classifier.result.plot.lift_curve()

        assert "Lift Curve - LogisticRegression" == ax.title._text
        assert "Lift" == ax.get_ylabel()
        assert "% of Data" == ax.get_xlabel()
        assert pytest.approx(19.5) == np.sum(ax.lines[0].get_xdata())
        assert pytest.approx(60.291, rel=0.0001) == np.sum(ax.lines[0].get_ydata())
        plt.close()

    def test_lift_chart_fails_correctly_with_2d_proba(self):
        x, y = load_iris(return_X_y=True)
        clf = LogisticRegression(solver="liblinear", multi_class="auto")
        clf.fit(x, y)
        proba = clf.predict_proba(x)
        with pytest.raises(VizError):
            plot_lift_curve(y, proba)


class TestPredictionErrorPlot:
    def test_prediction_error_plots_can_be_given_an_ax(self, regression: Model):
        fig, ax = plt.subplots()
        test_ax = regression.result.plot.prediction_error(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_prediction_error_plots_have_correct_data(self, regression: Model):
        ax = regression.result.plot.prediction_error()
        x, y = regression.result.plot._data.test_x, regression.result.plot._data.test_y
        y_pred = regression.result.model.estimator.predict(x)

        assert "Prediction Error - LinearRegression" == ax.title._text
        assert "$\\hat{y}$" == ax.get_ylabel()
        assert "$y$" == ax.get_xlabel()

        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y == ax.collections[0].get_offsets()[:, 0])
        plt.close()


class TestResidualPlot:
    def test_residuals_plots_can_be_given_an_ax(self, regression: Model):
        fig, ax = plt.subplots()
        test_ax = regression.result.plot.residuals(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_residual_plots_have_correct_data(self, regression: Model):
        ax = regression.result.plot.residuals()
        x, y = regression.result.plot._data.test_x, regression.result.plot._data.test_y
        y_pred = regression.result.model.estimator.predict(x)
        expected = y_pred - y

        assert "Residual Plot - LinearRegression" == ax.title._text
        assert "Residuals" == ax.get_ylabel()
        assert "Predicted Value" == ax.get_xlabel()

        assert np.all(expected == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 0])
        plt.close()


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

    def test_roc_curve_fails_correctly_without_predict_proba(
        self, test_dataset: Dataset
    ):
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(test_dataset)
        with pytest.raises(VizError):
            result.plot.roc_curve()


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

        assert "Precision-Recall - LogisticRegression" == ax.title._text
        assert "Precision" == ax.get_ylabel()
        assert "Recall" == ax.get_xlabel()
        assert np.all(recall == ax.lines[0].get_xdata())
        assert np.all(precision == ax.lines[0].get_ydata())
        plt.close()

    def test_pr_curve_fails_correctly_without_predict_proba(self, test_dataset: Model):
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(test_dataset)
        with pytest.raises(VizError):
            result.plot.pr_curve()
        plt.close()

    def test_pr_curve_can_use_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        x, y = classifier.result.plot._data.test_x, classifier.result.plot._data.test_y
        y_proba = classifier.estimator.predict_proba(x)[:, 1]

        assert ax is plot_pr_curve(y, y_proba, ax=ax)
        plt.close()
