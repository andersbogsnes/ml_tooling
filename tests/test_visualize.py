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
from sklearn.metrics import roc_curve, precision_recall_curve, get_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml_tooling import ModelData
from ml_tooling.metrics.permutation_importance import _get_feature_importance
from ml_tooling.metrics.permutation_importance import _permutation_importances
from ml_tooling.plots import plot_lift_curve, plot_confusion_matrix, plot_pr_curve
from ml_tooling.plots.utils import VizError
from ml_tooling.result.viz import RegressionVisualize, ClassificationVisualize
from ml_tooling.transformers import ToCategorical


class TestVisualize:
    def test_result_regression_gets_correct_visualizers(self, regression):
        result = regression.result
        assert isinstance(result.plot, RegressionVisualize)

    def test_result_classification_gets_correct_visualizers(self, classifier):
        result = classifier.result
        assert isinstance(result.plot, ClassificationVisualize)

    @pytest.mark.parametrize(
        "attr, option",
        [("residuals", None), ("prediction_error", None), ("feature_importance", 100)],
    )
    def test_regression_visualize_has_all_plots(self, attr, option, regression):
        result = regression.result.plot
        if option:
            plotter = getattr(result, attr)(option)
        else:
            plotter = getattr(result, attr)()
        assert isinstance(plotter, Axes)
        plt.close()

    @pytest.mark.parametrize(
        "attr, option",
        [
            ("confusion_matrix", None),
            ("roc_curve", None),
            ("lift_curve", None),
            ("feature_importance", 100),
            ("pr_curve", None),
        ],
    )
    def test_classifier_visualize_has_all_plots(self, attr, option, classifier):
        result = classifier.result.plot
        if option:
            plotter = getattr(result, attr)(option)
        else:
            plotter = getattr(result, attr)()
        assert isinstance(plotter, Axes)
        plt.close()


class TestConfusionMatrixPlot:
    def test_confusion_matrix_plots_have_correct_data(self, classifier):
        ax = classifier.result.plot.confusion_matrix()

        assert "Confusion Matrix - LogisticRegression - Normalized" == ax.title._text
        result = [text._text for text in ax.texts]
        assert pytest.approx(1) == np.round(np.sum([float(x) for x in result]), 1)
        assert {"0.63", "0.18", "0.11", "0.08"} == set(result)
        assert "True Label" == ax.get_ylabel()
        assert "Predicted Label" == ax.get_xlabel()
        plt.close()

    def test_confusion_matrix_plots_have_correct_data_when_not_normalized(
        self, classifier
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
    def test_feature_importance_plots_have_correct_data(self, classifier):
        ax = classifier.result.plot.feature_importance(samples=100)

        expected = {"0.09", "-0.04", "-0.04", "0.07"}
        assert expected == {text._text for text in ax.texts}
        assert "Feature Importance - LogisticRegression" == ax.title._text
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_have_no_labels_if_value_is_false(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(values=False, samples=100)
        assert 0 == len(ax.texts)
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        assert "Feature Importance - LogisticRegression" == ax.title._text
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_top_n_is_set(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(top_n=2, samples=100)
        assert 2 == len(ax.texts)
        assert {"0.09", "0.07"} == {text._text for text in ax.texts}
        assert "Feature Importance - LogisticRegression - Top 2" == ax.title._text
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_top_n_is_percent(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(top_n=0.2, samples=100)
        assert 1 == len(ax.texts)
        assert {"0.09"} == {text._text for text in ax.texts}
        assert "Feature Importance - LogisticRegression - Top 20%" == ax.title._text
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_bottom_n_is_int(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(bottom_n=2, samples=100)
        assert 2 == len(ax.texts)
        assert {"-0.04", "-0.04"} == {text._text for text in ax.texts}
        assert "Feature Importance - LogisticRegression - Bottom 2" == ax.title._text
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    @pytest.mark.xfail
    def test_feature_importance_plots_have_correct_labels_when_bottom_n_is_percent(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(bottom_n=0.2, samples=100)
        assert 1 == len(ax.texts)
        assert {"-0.04"} == {text._text for text in ax.texts}
        assert "Feature Importance - LogisticRegression - Bottom 20%" == ax.title._text
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_correct_if_top_n_is_int_and_bottom_n_is_int(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(top_n=1, bottom_n=1, samples=100)
        assert 2 == len(ax.texts)
        assert {"0.09", "-0.04"} == {text._text for text in ax.texts}
        assert (
            "Feature Importance - LogisticRegression - Top 1 - Bottom 1"
            == ax.title._text
        )
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_correct_when_top_n_is_int_and_bottom_n_is_percent(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(
            top_n=1, bottom_n=0.2, samples=100
        )
        assert 2 == len(ax.texts)
        assert {"0.09", "-0.04"} == {text._text for text in ax.texts}
        assert (
            "Feature Importance - LogisticRegression - Top 1 - Bottom 20%"
            == ax.title._text
        )
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_correct_when_top_n_is_percent_and_bottom_n_is_int(
        self, classifier
    ):
        ax = classifier.result.plot.feature_importance(
            top_n=0.2, bottom_n=1, samples=100
        )
        assert 2 == len(ax.texts)
        assert {"-0.04", "0.09"} == {text._text for text in ax.texts}
        assert (
            "Feature Importance - LogisticRegression - Top 20% - Bottom 1"
            == ax.title._text
        )
        assert "Features" == ax.get_ylabel()
        assert (
            "Importance:  Decrease in accuracy from baseline of 0.64" == ax.get_xlabel()
        )
        plt.close()

    def test_feature_importance_plots_correctly_in_pipeline(
        self, base, categorical, test_dataset
    ):
        pipe = Pipeline(
            [
                ("tocategory", ToCategorical()),
                ("clf", RandomForestClassifier(n_estimators=10)),
            ]
        )

        model = ModelData(pipe)
        result = model.score_estimator(test_dataset)
        ax = result.plot.feature_importance(samples=100)

        assert "Feature Importance - RandomForestClassifier" == ax.title._text
        assert 4 == len(ax.get_yticklabels())
        plt.close()


class TestLiftCurvePlot:
    def test_lift_curve_have_correct_data(self, classifier):
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
    def test_prediction_error_plots_have_correct_data(self, regression):
        ax = regression.result.plot.prediction_error()
        x, y = regression.result.plot._data.test_x, regression.result.plot._data.test_y
        y_pred = regression.result.estimator.predict(x)

        assert "Prediction Error - LinearRegression" == ax.title._text
        assert "$\\hat{y}$" == ax.get_ylabel()
        assert "$y$" == ax.get_xlabel()

        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y == ax.collections[0].get_offsets()[:, 0])
        plt.close()


class TestResidualPlot:
    def test_residual_plots_have_correct_data(self, regression):
        ax = regression.result.plot.residuals()
        x, y = regression.result.plot._data.test_x, regression.result.plot._data.test_y
        y_pred = regression.result.estimator.predict(x)
        expected = y_pred - y

        assert "Residual Plot - LinearRegression" == ax.title._text
        assert "Residuals" == ax.get_ylabel()
        assert "Predicted Value" == ax.get_xlabel()

        assert np.all(expected == ax.collections[0].get_offsets()[:, 1])
        assert np.all(y_pred == ax.collections[0].get_offsets()[:, 0])
        plt.close()


class TestRocCurve:
    def test_roc_curve_have_correct_data(self, classifier):
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

    def test_roc_curve_fails_correctly_without_predict_proba(self, base, test_dataset):
        svc = base(SVC(gamma="scale"))
        result = svc.score_estimator(test_dataset)
        with pytest.raises(VizError):
            result.plot.roc_curve()


class TestPRCurve:
    def test_pr_curve_have_correct_data(self, classifier):
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

    def test_pr_curve_fails_correctly_without_predict_proba(self, base, test_dataset):
        svc = base(SVC(gamma="scale"))
        result = svc.score_estimator(test_dataset)
        with pytest.raises(VizError):
            result.plot.pr_curve()
        plt.close()

    def test_pr_curve_can_use_ax(self, classifier):
        fig, ax = plt.subplots()
        x, y = classifier.result.plot._data.test_x, classifier.result.plot._data.test_y
        y_proba = classifier.estimator.predict_proba(x)[:, 1]

        assert ax is plot_pr_curve(y, y_proba, ax=ax)
        plt.close()


class TestGetFeatureImportance:
    def test_viz_get_feature_importance_regression_returns_importance(self, regression):
        sample = 10
        importance, baseline = _get_feature_importance(regression.result.plot, sample)

        model = regression.result.plot._estimator
        metric = get_scorer(regression.result.plot._config.REGRESSION_METRIC)
        train_x = regression.result.plot._data.train_x
        train_y = regression.result.plot._data.train_y

        expected_importance, expected_baseline = _permutation_importances(
            model, metric, train_x, train_y, sample
        )

        assert np.all(expected_baseline == baseline)
        assert np.all(expected_importance == importance)

    def test_get_feature_importance_returns_importance_from_regression_pipeline(
        self, base, pipeline_linear, test_dataset
    ):
        pipe = base(pipeline_linear)
        pipe.score_estimator(test_dataset)
        sample = 10
        importance, baseline = _get_feature_importance(pipe.result.plot, sample)

        model = pipe.result.plot._estimator
        metric = get_scorer(pipe.result.plot._config.REGRESSION_METRIC)
        train_x = pipe.result.plot._data.train_x
        train_y = pipe.result.plot._data.train_y
        expected_importance, expected_baseline = _permutation_importances(
            model, metric, train_x, train_y, sample
        )

        assert np.all(expected_baseline == baseline)
        assert np.all(expected_importance == importance)

    def test_viz_get_feature_importance_returns_feature_importance_from_classifier(
        self, base, test_dataset
    ):
        classifier = base(RandomForestClassifier(n_estimators=10))
        result = classifier.score_estimator(test_dataset)
        sample = 10
        importance, baseline = _get_feature_importance(result.plot, sample)

        model = classifier.result.plot._estimator
        metric = get_scorer(classifier.result.plot._config.CLASSIFIER_METRIC)
        train_x = classifier.result.plot._data.train_x
        train_y = classifier.result.plot._data.train_y
        expected_importance, expected_baseline = _permutation_importances(
            model, metric, train_x, train_y, sample
        )

        assert np.all(expected_baseline == baseline)
        assert np.all(expected_importance == importance)
