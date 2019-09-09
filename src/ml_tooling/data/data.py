import abc
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from ml_tooling.utils import DataType
from sklearn.utils import indexable


class TrainingDataSet:
    _x: Optional[DataType] = None
    _y: Optional[DataType] = None
    _test_x: Optional[DataType] = None
    _test_y: Optional[DataType] = None
    _train_y: Optional[DataType] = None
    _train_x: Optional[DataType] = None

    def create_train_test(
        self, stratify=None, shuffle=True, test_size=0.25, seed=42
    ) -> "TrainingDataSet":
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

        self._train_x, self._test_x, self._train_y, self._test_y = train_test_split(
            self.x,
            self.y,
            stratify=stratify,
            shuffle=shuffle,
            test_size=test_size,
            random_state=seed,
        )
        return self

    @property
    def test_x(self):
        if self._test_x is None:
            self.create_train_test()
        return self._test_x

    @property
    def test_y(self):
        if self._test_y is None:
            self.create_train_test()
        return self._test_y

    @property
    def train_x(self):
        if self._train_x is None:
            self.create_train_test()
        return self._train_x

    @property
    def train_y(self):
        if self._train_y is None:
            self.create_train_test()
        return self._train_y

    @property
    def x(self):
        if self._x is None:
            self._x, self._y = indexable(self.load())
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._x, self._y = indexable(self.load())
        return self._y

    @abc.abstractmethod
    def load(self, *args, **kwargs) -> Tuple[DataType, DataType]:
        raise NotImplementedError


class PredictionDataSet:
    @abc.abstractmethod
    def load(self, *args, **kwargs):
        raise NotImplementedError
