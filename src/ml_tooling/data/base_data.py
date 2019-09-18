import abc
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from ml_tooling.utils import DataType
from sklearn.utils import indexable


class DataSet:
    _x: Optional[DataType] = None
    _y: Optional[DataType] = None
    test_x: Optional[DataType] = None
    test_y: Optional[DataType] = None
    train_y: Optional[DataType] = None
    train_x: Optional[DataType] = None

    def create_train_test(
        self, stratify=None, shuffle=True, test_size=0.25, seed=42
    ) -> "DataSet":
        """
        Creates a training and testing dataset and storing it on the data object.
        :param stratify:
            What to stratify the split on. Usually y if given a classification problem
        :param shuffle:
            Whether or not to shuffle the data
        :param test_size:
            What percentage of the data will be part of the test set
         :param seed:
            Random seed for train_test_split
        :return:
            self
        """

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.x,
            self.y,
            stratify=stratify,
            shuffle=shuffle,
            test_size=test_size,
            random_state=seed,
        )
        return self

    @property
    def x(self):
        if self._x is None:
            self._x, self._y = indexable(self.load_training_data())
        return self._x

    @x.setter
    def x(self, data):
        self._x = data

    @property
    def y(self):
        if self._y is None:
            self._x, self._y = indexable(self.load_training_data())
        return self._y

    @y.setter
    def y(self, data):
        self._y = data

    @abc.abstractmethod
    def load_training_data(self, *args, **kwargs) -> Tuple[DataType, DataType]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_prediction_data(self, *args, **kwargs) -> DataType:
        raise NotImplementedError
