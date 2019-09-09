from sklearn.model_selection import train_test_split
from sklearn.utils import indexable

from ml_tooling.utils import DataType


class Data:
    """
    Container for storing data. Contains both x and y, while also handling train_test_split
    """

    def __init__(self, x: DataType, y: DataType):
        self.x, self.y = indexable(x, y)
        self.train_y = None
        self.train_x = None
        self.test_y = None
        self.test_x = None

    def create_train_test(
        self, stratify=None, shuffle=True, test_size=0.25, seed=42
    ) -> "Data":
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

    @classmethod
    def with_train_test(
        cls,
        x: DataType,
        y: DataType,
        stratify=None,
        shuffle=True,
        test_size=0.25,
        seed=42,
    ) -> "Data":
        """
        Creates a new instance of Data with train and test already instantiated
        :param x:
            Features
        :param y:
            Target
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
        instance = cls(x, y)
        return instance.create_train_test(
            stratify=stratify, shuffle=shuffle, test_size=test_size, seed=seed
        )
