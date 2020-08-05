from typing import List

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset, load_demo_dataset
from ml_tooling.plots.permutation_importance import plot_permutation_importance
from ml_tooling.transformers import DFStandardScaler


class TestPermutationImportancePlot:
    @pytest.fixture()
    def feature_importances(self):
        return ["0.09", "0.07", "0.07", "-0.02"]

    @pytest.fixture()
    def ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance()
        plt.close()

    @pytest.fixture()
    def no_label_ax(self, classifier: Model):
        yield classifier.result.plot.permutation_importance(add_label=False)
        plt.close()

    @pytest.fixture()
    def top_n_ax(self, classifier: Model):
        yield classifier.result.plot.permutation_importance(top_n=2)
        plt.close()

    @pytest.fixture()
    def top_n_percent_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance(top_n=0.2)
        plt.close()

    @pytest.fixture()
    def bottom_n_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance(bottom_n=2)
        plt.close()

    @pytest.fixture()
    def bottom_n_percent_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance(bottom_n=0.2)
        plt.close()

    @pytest.fixture()
    def top_bottom_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance(top_n=1, bottom_n=1)
        plt.close()

    @pytest.fixture()
    def top_percent_bottom_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance(top_n=0.2, bottom_n=1)
        plt.close()

    @pytest.fixture()
    def top_bottom_percent_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.permutation_importance(top_n=1, bottom_n=0.2)
        plt.close()

    def test_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.permutation_importance(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_correct_data(
        self, classifier: Model, ax: Axes, feature_importances: List[str]
    ):
        assert {text.get_text() for text in ax.texts} == set(feature_importances)

    def test_has_correct_ylabel(self, ax: Axes):
        assert ax.get_ylabel() == "Feature Labels"

    def test_has_correct_xlabel(self, ax: Axes):
        assert (
            ax.get_xlabel()
            == "Permuted Feature Importance (Accuracy) Relative to Baseline"
        )

    def test_has_correct_title(self, ax: Axes):
        assert (
            ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression"
        )

    def test_no_labels_if_add_label_false(self, no_label_ax):
        assert len(no_label_ax.texts) == 0

    def test_has_two_labels_when_top_n_is_set(self, top_n_ax: Axes):
        assert 2 == len(top_n_ax.texts)

    def test_has_correct_data_when_top_n_is_set(
        self, top_n_ax: Axes, feature_importances
    ):
        assert {text.get_text() for text in top_n_ax.texts} == set(
            feature_importances[:2]
        )

    def test_has_correct_title_when_top_n_is_set(self, top_n_ax: Axes):
        assert (
            top_n_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Top 2"
        )

    def test_has_correct_labels_when_top_n_is_percent(
        self, top_n_percent_ax: Axes, feature_importances: List[str]
    ):
        assert {text.get_text() for text in top_n_percent_ax.texts} == {
            feature_importances[0]
        }

    def test_has_correct_title_when_top_n_is_percent(self, top_n_percent_ax: Axes):
        assert (
            top_n_percent_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Top 20%"
        )

    def test_has_correct_labels_when_bottom_n_is_int(
        self, bottom_n_ax: Axes, feature_importances
    ):
        assert {text.get_text() for text in bottom_n_ax.texts} == set(
            feature_importances[-2:]
        )

    def test_has_correct_title_when_bottom_n_is_int(self, bottom_n_ax: Axes):
        assert (
            bottom_n_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Bottom 2"
        )

    def test_has_correct_labels_when_bottom_n_is_percent(
        self, bottom_n_percent_ax: Axes, feature_importances: List[str]
    ):
        assert {text.get_text() for text in bottom_n_percent_ax.texts} == {
            feature_importances[-1]
        }

    def test_has_correct_title_when_bottom_n_is_percent(
        self, bottom_n_percent_ax: Axes
    ):
        assert (
            bottom_n_percent_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Bottom 20%"
        )

    def test_has_correct_labels_if_top_n_is_int_and_bottom_n_is_int(
        self, top_bottom_ax: Axes, feature_importances: List[str]
    ):
        assert {text.get_text() for text in top_bottom_ax.texts} == {
            feature_importances[0],
            feature_importances[-1],
        }

    def test_has_correct_title_when_top_n_and_bottom_n_is_int(
        self, top_bottom_ax: Axes
    ):
        assert (
            top_bottom_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Top 1 - Bottom 1"
        )

    def test_has_correct_labels_when_top_n_is_int_and_bottom_n_is_percent(
        self, top_bottom_percent_ax: Axes, feature_importances
    ):
        assert {text.get_text() for text in top_bottom_percent_ax.texts} == {
            feature_importances[0],
            feature_importances[-1],
        }

    def test_has_correct_title_when_top_n_is_int_and_bottom_n_is_percent(
        self, top_bottom_percent_ax
    ):
        assert (
            top_bottom_percent_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Top 1 - Bottom 20%"
        )

    def test_has_correct_labels_when_top_n_is_percent_and_bottom_n_is_int(
        self, top_percent_bottom_ax: Axes, feature_importances: List[str]
    ):
        assert {text.get_text() for text in top_percent_bottom_ax.texts} == {
            feature_importances[0],
            feature_importances[-1],
        }

    def test_has_correct_title_when_top_n_is_percent_and_bottom_n_is_int(
        self, top_percent_bottom_ax: Axes
    ):
        assert (
            top_percent_bottom_ax.title.get_text()
            == "Permutation Importances (Accuracy) - LogisticRegression - Top 20% - Bottom 1"
        )

    def test_plots_correctly_in_pipeline(self, train_iris_dataset: Dataset):
        pipe = Pipeline(
            [
                ("scale", DFStandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=10)),
            ]
        )

        model = Model(pipe)
        result = model.score_estimator(train_iris_dataset)
        ax = result.plot.permutation_importance()

        assert (
            ax.title.get_text()
            == "Permutation Importances (Accuracy) - RandomForestClassifier"
        )

        assert 4 == len(list(ax.get_yticklabels()))
        plt.close()

    def test_doesnt_error_in_on_large_datasets(self, train_iris_dataset: Dataset):
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
        assert result.plot.permutation_importance()
        plt.close()

    def test_regressor_works_as_expected(self, regression: Model):
        ax = regression.result.plot.permutation_importance()
        assert ax.title.get_text() == "Permutation Importances (R2) - LinearRegression"
        plt.close()

    def test_can_use_different_scoring_metrics(self, classifier: Model):
        ax = classifier.result.plot.permutation_importance(scoring="roc_auc")
        assert (
            ax.title.get_text()
            == "Permutation Importances (Roc_Auc) - LogisticRegression"
        )
        plt.close()

    @pytest.mark.parametrize(
        "scorer, model",
        [("r2", RandomForestRegressor()), ("accuracy", RandomForestClassifier())],
    )
    def test_function_has_correct_scorer_when_passing_classifier_and_regressor(
        self, scorer: str, model: BaseEstimator
    ):
        dataset = load_demo_dataset("iris")
        model.fit(dataset.x, dataset.y)
        ax = plot_permutation_importance(model, dataset.x, dataset.y, n_repeats=1)
        assert ax.title.get_text() == f"Permutation Importances ({scorer.title()})"


def test_doesnt_error_in_on_large_datasets(train_iris_dataset: Dataset):

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
    plt.close()
