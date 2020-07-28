from typing import List

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ml_tooling import Model
from ml_tooling.data import Dataset
from ml_tooling.transformers import DFStandardScaler
from ml_tooling.utils import VizError


class TestFeatureImportancePlot:
    @pytest.fixture()
    def feature_importances(self):
        return ["-1.51", "-1.24", "0.58", "0.38"]

    @pytest.fixture()
    def ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance()
        plt.close()

    @pytest.fixture()
    def no_label_ax(self, classifier: Model):
        yield classifier.result.plot.feature_importance(add_label=False)
        plt.close()

    @pytest.fixture()
    def top_n_ax(self, classifier: Model):
        yield classifier.result.plot.feature_importance(top_n=2)
        plt.close()

    @pytest.fixture()
    def top_n_percent_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance(top_n=0.2)
        plt.close()

    @pytest.fixture()
    def bottom_n_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance(bottom_n=2)
        plt.close()

    @pytest.fixture()
    def bottom_n_percent_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance(bottom_n=0.2)
        plt.close()

    @pytest.fixture()
    def top_bottom_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance(top_n=1, bottom_n=1)
        plt.close()

    @pytest.fixture()
    def top_percent_bottom_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance(top_n=0.2, bottom_n=1)
        plt.close()

    @pytest.fixture()
    def top_bottom_percent_ax(self, classifier: Model) -> Axes:
        yield classifier.result.plot.feature_importance(top_n=1, bottom_n=0.2)
        plt.close()

    def test_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.feature_importance(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_correct_data(
        self, classifier: Model, ax: Axes, feature_importances: List[str]
    ):
        assert {text.get_text() for text in ax.texts} == set(feature_importances)

    def test_has_correct_ylabel(self, ax: Axes):
        assert ax.get_ylabel() == "Feature Labels"

    def test_has_correct_xlabel(self, ax: Axes):
        assert ax.get_xlabel() == "Coefficients"

    def test_has_correct_title(self, ax: Axes):
        assert ax.title.get_text() == "Feature Importances - LogisticRegression"

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
            == "Feature Importances - LogisticRegression - Top 2"
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
            == "Feature Importances - LogisticRegression - Top 20%"
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
            == "Feature Importances - LogisticRegression - Bottom 2"
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
            == "Feature Importances - LogisticRegression - Bottom 20%"
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
            == "Feature Importances - LogisticRegression - Top 1 - Bottom 1"
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
            == "Feature Importances - LogisticRegression - Top 1 - Bottom 20%"
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
            == "Feature Importances - LogisticRegression - Top 20% - Bottom 1"
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
        ax = result.plot.feature_importance()

        assert ax.title.get_text() == "Feature Importances - RandomForestClassifier"

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
        assert result.plot.feature_importance()
        plt.close()

    def test_regressor_works_as_expected(self, regression: Model):
        ax = regression.result.plot.feature_importance()
        assert ax.title.get_text() == "Feature Importances - LinearRegression"
        plt.close()

    def test_has_correct_xlabel_when_using_trees(self, train_iris_dataset: Dataset):
        model = Model(RandomForestClassifier())
        result = model.score_estimator(train_iris_dataset)
        ax = result.plot.feature_importance()
        assert ax.get_xlabel() == "Feature Importances"

    def test_raises_if_passed_model_without_feature_importance_or_coefs(
        self, train_iris_dataset: Dataset
    ):
        model = Model(KNeighborsClassifier())
        result = model.score_estimator(train_iris_dataset)
        with pytest.raises(VizError):
            result.plot.feature_importance()
