from matplotlib import pyplot as plt

from ml_tooling import Model


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
        plt.close()

    def test_validation_curve_plot_can_multiprocess(self, classifier: Model):
        param_range = [0.001, 0.01, 0.01, 0.1, 1]
        assert classifier.result.plot.validation_curve(
            param_name="C", param_range=param_range, n_jobs=-1
        )
        plt.close()

    def test_validation_curve_can_plot_other_metrics(self, classifier: Model):
        param_range = [0.001, 0.01, 0.01, 0.1, 1]
        test_ax = classifier.result.plot.validation_curve(
            param_name="C", param_range=param_range, scoring="roc_auc"
        )

        assert test_ax.get_ylabel() == "Roc_Auc Score"
        assert test_ax.get_legend().texts[0].get_text() == "Training Roc_Auc"
        assert test_ax.get_legend().texts[1].get_text() == "Test Roc_Auc"
        plt.close()
