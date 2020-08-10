from matplotlib import pyplot as plt

from ml_tooling import Model


class TestLearningCurve:
    def test_learning_curve_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.learning_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_learning_curve_plots_have_correct_elements(self, classifier: Model):
        test_ax = classifier.result.plot.learning_curve(cv=5)
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
        plt.close()

    def test_learning_curve_can_use_different_scoring_parameters(
        self, classifier: Model
    ):
        test_ax = classifier.result.plot.learning_curve(scoring="roc_auc")
        assert test_ax.get_ylabel() == "Roc_Auc Score"
        assert test_ax.get_legend().texts[0].get_text() == "Training Roc_Auc"
        assert test_ax.get_legend().texts[1].get_text() == "Cross-validated Roc_Auc"
        plt.close()
