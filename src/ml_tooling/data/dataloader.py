import abc
from typing import Optional

from sklearn.model_selection import train_test_split
from ml_tooling.utils import DataType
from sklearn.utils import indexable


class DataSet:
    def __init__(self, x: DataType, y: DataType):
        self.x: Optional[DataType] = indexable(x)
        self.y: Optional[DataType] = indexable(y)
        self.test_x: Optional[DataType] = None
        self.test_y: Optional[DataType] = None
        self.train_y: Optional[DataType] = None
        self.train_x: Optional[DataType] = None

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

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError
