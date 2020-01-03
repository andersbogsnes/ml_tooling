"""
Test file for visualisations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from ml_tooling.plots import (
    plot_lift_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_feature_importance,
)
from ml_tooling.utils import VizError
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
        "attr",
        ["residuals", "prediction_error", "feature_importance", "learning_curve"],
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
            "learning_curve",
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
        ax = classifier.result.plot.feature_importance(random_state=42)

        expected = {"0.04", "0.08", "-0.03", "0.02"}
        assert {text._text for text in ax.texts} == expected
        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text() == "Feature Importances (Accuracy) - LogisticRegression"
        )
        plt.close()

    def test_feature_importance_plots_have_no_labels_if_value_is_false(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(add_label=False)
        assert len(ax.texts) == 0
        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text() == "Feature Importances (Accuracy) - LogisticRegression"
        )
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_top_n_is_set(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=2, random_state=42)
        assert 2 == len(ax.texts)
        assert {text.get_text() for text in ax.texts} == {"0.04", "0.08"}

        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Top 2"
        )
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_top_n_is_percent(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(top_n=0.2, random_state=42)
        assert len(ax.texts) == 1
        assert {text.get_text() for text in ax.texts} == {"0.08"}

        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Top 20%"
        )
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_bottom_n_is_int(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(bottom_n=2, random_state=42)
        assert len(ax.texts) == 2
        assert {text.get_text() for text in ax.texts} == {"0.02", "-0.03"}

        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Bottom 2"
        )
        plt.close()

    def test_feature_importance_plots_have_correct_labels_when_bottom_n_is_percent(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(bottom_n=0.2, random_state=42)
        assert len(ax.texts) == 1
        assert {text.get_text() for text in ax.texts} == {"0.02"}

        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Bottom 20%"
        )
        plt.close()

    def test_feature_importance_plots_correct_if_top_n_is_int_and_bottom_n_is_int(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(
            top_n=1, bottom_n=1, random_state=42
        )
        assert len(ax.texts) == 2
        assert {text.get_text() for text in ax.texts} == {"0.08", "0.02"}
        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Top 1 - Bottom 1"
        )
        plt.close()

    def test_feature_importance_plots_correct_when_top_n_is_int_and_bottom_n_is_percent(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(
            top_n=1, bottom_n=0.2, random_state=42
        )
        assert 2 == len(ax.texts)
        assert {text.get_text() for text in ax.texts} == {"0.08", "0.02"}
        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Top 1 - Bottom 20%"
        )
        plt.close()

    def test_feature_importance_plots_correct_when_top_n_is_percent_and_bottom_n_is_int(
        self, classifier: Model
    ):
        ax = classifier.result.plot.feature_importance(
            top_n=0.2, bottom_n=1, random_state=42
        )
        assert len(ax.texts) == 2
        assert {text.get_text() for text in ax.texts} == {"0.02", "0.08"}
        assert ax.get_ylabel() == "Feature Labels"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )
        assert (
            ax.title.get_text()
            == "Feature Importances (Accuracy) - LogisticRegression - Top 20% - Bottom 1"
        )
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

        assert (
            "Feature Importances (Accuracy) - RandomForestClassifier" == ax.title._text
        )
        assert 4 == len(list(ax.get_yticklabels()))
        plt.close()

    def test_feature_importance_doesnt_error_in_on_large_datasets(
        self, test_dataset: Dataset
    ):
        """
        When joblib.Parallel receives data that is larger than a given size, it will do a read-only
        memmap on the data.

        scikit-learn's permutation importance implementation modifies the object,
        resulting in an error.

        This test replicates the error by creating a large DataFrame
        """

        # Make a new dataset with lots of rows to trigger the joblib.Parallel error
        class IrisData(Dataset):
            def load_training_data(self):
                data = load_iris()
                return (
                    pd.DataFrame(
                        data=data.data.repeat(1000, axis=0), columns=data.feature_names
                    ),
                    data.target.repeat(1000),
                )

            def load_prediction_data(self):
                pass

        data = IrisData().create_train_test()
        result = Model(RandomForestClassifier(n_estimators=2)).score_estimator(data)
        assert result.plot.feature_importance()

    def test_can_use_different_scoring_metrics(self, classifier: Model):
        ax = classifier.result.plot.feature_importance(
            scoring="roc_auc", random_state=42
        )
        assert (
            ax.title.get_text() == "Feature Importances (Roc_Auc) - LogisticRegression"
        )
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Roc_Auc) Relative to Baseline"
        )

    def test_can_use_feature_importance_with_regressor(self, regression: Model):
        ax = regression.result.plot.feature_importance()
        assert ax.title.get_text() == "Feature Importances (R2) - LinearRegression"
        assert (
            ax.get_xlabel() == "Permuted Feature Importance (R2) Relative to Baseline"
        )

    def test_plot_feature_importance_with_default_metrics(self, classifier: Model):
        ax = plot_feature_importance(
            classifier.estimator, classifier.result.data.x, classifier.result.data.y
        )

        assert ax.title.get_text() == "Feature Importances (Accuracy)"
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )


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

        assert "Precision-Recall - LogisticRegression" == ax.title.get_text()
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


class TestLearningCurve:
    def test_learning_curve_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.learning_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_learning_curve_plots_have_correct_elements(self, classifier: Model):
        test_ax = classifier.result.plot.learning_curve()
        assert test_ax.title.get_text() == "Learning Curve - LogisticRegression"
        assert test_ax.get_ylabel() == "Accuracy Score"
        assert test_ax.get_xlabel() == "Number of Examples Used"
        assert test_ax.get_legend().texts[0].get_text() == "Training Accuracy"
        assert test_ax.get_legend().texts[1].get_text() == "Cross-validated Accuracy"
        # We have 5 CV folds, so 4/5ths of the data will be used
        assert (
            test_ax.lines[0].get_xdata().max()
            == (len(classifier.result.data.train_x) * 4) // 5
        )

    def test_learning_curve_can_use_different_scoring_parameters(
        self, classifier: Model
    ):
        test_ax = classifier.result.plot.learning_curve(scoring="roc_auc")
        assert test_ax.get_ylabel() == "Roc_Auc Score"
        assert test_ax.get_legend().texts[0].get_text() == "Training Roc_Auc"
        assert test_ax.get_legend().texts[1].get_text() == "Cross-validated Roc_Auc"


class TestValidationCurve:
    def test_validation_curve_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.validation_curve(
            param_name="C", param_range=[0.1, 0.001], ax=ax
        )
        assert ax == test_ax
        plt.close()

    def test_validation_curve_plot_has_correct_attributes(self, classifier: Model):
        param_range = [0.001, 0.01, 0.01, 0.1, 1]
        test_ax = classifier.result.plot.validation_curve(
            param_name="C", param_range=param_range
        )
        assert test_ax.get_title() == "Validation Curve - LogisticRegression"
        assert test_ax.get_ylabel() == "Accuracy Score"
        assert test_ax.get_xlabel() == "C"
        assert test_ax.lines[0].get_xdata().max() == 1
        assert test_ax.lines[0].get_xdata().min() == 0.001
        assert test_ax.get_legend().texts[0].get_text() == "Training Accuracy"
        assert test_ax.get_legend().texts[1].get_text() == "Test Accuracy"

    def test_validation_curve_plot_can_multiprocess(self, classifier: Model):
        param_range = [0.001, 0.01, 0.01, 0.1, 1]
        assert classifier.result.plot.validation_curve(
            param_name="C", param_range=param_range, n_jobs=-1
        )

    def test_validation_curve_can_plot_other_metrics(self, classifier: Model):
        param_range = [0.001, 0.01, 0.01, 0.1, 1]
        test_ax = classifier.result.plot.validation_curve(
            param_name="C", param_range=param_range, scoring="roc_auc"
        )

        assert test_ax.get_ylabel() == "Roc_Auc Score"
        assert test_ax.get_legend().texts[0].get_text() == "Training Roc_Auc"
        assert test_ax.get_legend().texts[1].get_text() == "Test Roc_Auc"
