import abc
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from ml_tooling.utils import DataType, DatasetError
from sklearn.utils import indexable
from ml_tooling.plots.viz import DataVisualize


class Dataset(metaclass=abc.ABCMeta):
    """
    Baseclass for creating Datasets.
    Subclass Dataset and provide a :meth:`load_training_data`
    and :meth:`load_prediction_data` method
    """

    _x: Optional[pd.DataFrame] = None
    _y: Optional[DataType] = None
    test_x: Optional[pd.DataFrame] = None
    test_y: Optional[DataType] = None
    train_y: Optional[DataType] = None
    train_x: Optional[pd.DataFrame] = None
    cached_data: Optional[pd.DataFrame] = None

    @property
    def plot(self):
        return DataVisualize(self)

    def create_train_test(
        self,
        stratify: bool = False,
        shuffle: bool = True,
        test_size: float = 0.25,
        seed: int = 42,
    ) -> "Dataset":
        """
        Creates a training and testing dataset and storing it on the data object.

        Parameters
        ----------

        stratify: DataType, optional
            What to stratify the split on. Usually y if given a classification problem
        shuffle:
            Whether or not to shuffle the data
        test_size:
            What percentage of the data will be part of the test set
        seed:
            Random seed for train_test_split

        Returns
        -------
        self
        """
        if self.y is None:
            raise DatasetError(
                "The dataset does not define a y value"
                " - cannot create a train-test split"
            )

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.x,
            self.y,
            stratify=self.y if stratify else None,
            shuffle=shuffle,
            test_size=test_size,
            random_state=seed,
        )
        return self

    @property
    def x(self):
        if self._x is None:
            self._x, self._y = indexable(*self._load_training_data())
        return self._x

    @x.setter
    def x(self, data):
        raise DatasetError("Trying to modify x - x is immutable")

    @property
    def y(self):
        if self._y is None:
            self._x, self._y = indexable(*self._load_training_data())
        return self._y

    @y.setter
    def y(self, data):
        raise DatasetError("Trying to modify y - y is immutable")

    @property
    def has_validation_set(self):
        if (
            self.test_x is None
            or self.train_x is None
            or self.test_y is None
            or self.train_y is None
        ):
            return False
        return True

    @property
    def class_name(self):
        return self.__class__.__name__

    def _load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
        x, y = self.load_training_data(*args, **kwargs)
        if x.empty:
            raise DatasetError("An empty dataset was returned by load_training_data")

        return x, y

    def _load_prediction_data(self, *args, **kwargs) -> pd.DataFrame:
        pred_data = self.load_prediction_data(*args, **kwargs)
        if pred_data.empty:
            raise DatasetError("An empty dataset was returned by load_prediction_data")
        self.cached_data = pred_data
        return pred_data

    @abc.abstractmethod
    def load_training_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, DataType]:
        """Abstract method to be implemented by user.
        Defines data to be used at training time where X is a dataframe and y is a numpy array

        Returns
        -------
        x, y: Tuple of DataTypes
            Training data to be used by the models
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_prediction_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Abstract method to be implemented by the user.
        Defines data to be used at prediction time, defined as a DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame of input features to get a prediction
        """
        raise NotImplementedError

    def _dump_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _load_data(self, data):
        raise NotImplementedError

    def copy_to(self, target: "Dataset"):
        data = self._dump_data()
        target._load_data(data)
        return target

    def __repr__(self):
        return f"<{self.__class__.__name__} - Dataset>"
