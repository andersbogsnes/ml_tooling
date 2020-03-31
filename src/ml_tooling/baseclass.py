import datetime
import pathlib
import joblib
import pandas as pd
from contextlib import contextmanager
from importlib.resources import path as import_path
from typing import Tuple, Optional, Sequence, Union, List, Iterable, Any
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import check_cv

from ml_tooling.config import DefaultConfig, ConfigGetter
from ml_tooling.data.base_data import Dataset
from ml_tooling.logging.logger import create_logger
from ml_tooling.metrics.metric import Metrics
from ml_tooling.result import ResultType
from ml_tooling.result.result import Result
from ml_tooling.result.result_group import ResultGroup
from ml_tooling.search.gridsearch import prepare_gridsearch_estimators
from ml_tooling.search.randomsearch import prepare_randomsearch_estimators
from ml_tooling.storage.base import Storage
from ml_tooling.utils import (
    MLToolingError,
    _validate_estimator,
    DatasetError,
    Estimator,
    is_pipeline,
    serialize_pipeline,
    _get_estimator_name,
    make_pipeline_from_definition,
    read_yaml,
    _classify,
)

logger = create_logger("ml_tooling")


class Model:
    """
    Wrapper class for Estimators
    """

    _config = None
    config = ConfigGetter()

    def __init__(self, estimator):
        self.estimator: Estimator = _validate_estimator(estimator)
        self.result: Optional[ResultType] = None

    @property
    def is_classifier(self) -> bool:
        return is_classifier(self.estimator)

    @property
    def is_regressor(self) -> bool:
        return is_regressor(self.estimator)

    @property
    def is_pipeline(self) -> bool:
        return is_pipeline(self.estimator)

    @property
    def estimator_name(self) -> str:
        return _get_estimator_name(self.estimator)

    @property
    def default_metric(self) -> str:
        """
        Defines default metric based on whether or not the estimator is a regressor or classifier.
        Then :attr:`~ml_tooling.config.DefaultConfig.CLASSIFIER_METRIC` or
        :attr:`~ml_tooling.config.DefaultConfig.CLASSIFIER_METRIC` is returned.

        Returns
        -------
        str
            Name of the metric
        """

        return (
            self.config.CLASSIFIER_METRIC
            if self.is_classifier
            else self.config.REGRESSION_METRIC
        )

    @default_metric.setter
    def default_metric(self, metric: str):
        if self.is_classifier:
            self.config.CLASSIFIER_METRIC = metric
        else:
            self.config.REGRESSION_METRIC = metric

    @staticmethod
    def list_estimators(storage: Storage) -> List[pathlib.Path]:
        """
        Gets a list of estimators from the given Storage

        Parameters
        ----------
        storage: Storage
            Storage class to list the estimators with

        Example
        -------
            storage = FileStorage('path/to/estimators_dir')
            estimator_list = Model.list_estimators(storage)

        Returns
        -------
        List[pathlib.Path]
            list of Paths
        """
        return storage.get_list()

    @classmethod
    def load_estimator(cls, storage: Storage, path: pathlib.Path) -> "Model":
        """
        Instantiates the class with a joblib pickled estimator.

        Parameters
        ----------
        storage : Storage
            Storage class to load the estimator with
        path: str, optional
            Path to estimator pickle file

        Example
        -------
        We can load a trained estimator from disk::

            storage = FileStorage('path/to/dir')
            my_estimator = Model.load_estimator(storage, 'my_model.pkl')

        We now have a trained estimator loaded.

        Returns
        -------
        Model
            Instance of Model with a saved estimator
        """
        filename = pathlib.Path(path).name
        estimator = storage.load(filename)
        instance = cls(estimator)
        logger.info(f"Loaded {instance.estimator_name}")
        return instance

    def save_estimator(self, storage: Storage = None, prod=False) -> pathlib.Path:
        """
        Saves the estimator as a binary file.

        Parameters
        ----------
        storage: Storage
            Storage class to save the estimator with

        prod: bool
            Whether this is a production model to be saved

        Example
        -------

        If we have trained an estimator and we want to save it to disk we can write::

            storage = FileStorage('/path/to/save/dir')
            model = Model(LinearRegression())
            saved_filename = model.save_estimator(storage)

        to save in the given folder.

        Returns
        -------
        pathlib.Path
            The path to where the estimator file was saved
        """
        if storage is None:
            storage = self.config.default_storage

        if prod:
            file_name = "production_model.pkl"
        else:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
            file_name = f"{self.estimator_name}_{now_str}.pkl"
        estimator_file = storage.save(self.estimator, file_name, prod=prod)

        logger.debug(f"Attempting to save estimator {estimator_file}")

        if self.config.LOG:
            if self.result is None:
                raise MLToolingError(
                    "You haven't scored the estimator - no results available to log"
                )

            self.result.log(
                saved_estimator_path=estimator_file, savedir=self.config.RUN_DIR
            )

        logger.info(f"Saved estimator to {estimator_file}")

        return estimator_file

    def to_dict(self) -> List[dict]:
        """
        Serializes the estimator to a dictionary

        Returns
        -------
        List of dicts
        """
        if self.is_pipeline:
            return serialize_pipeline(self.estimator)

        return [
            {
                "module": self.estimator.__class__.__module__,
                "classname": self.estimator.__class__.__name__,
                "params": self.estimator.get_params(),
            }
        ]

    @classmethod
    def from_yaml(cls, log_file: pathlib.Path) -> "Model":
        definitions = read_yaml(log_file)["estimator"]
        pipeline = make_pipeline_from_definition(definitions)

        return cls(pipeline)

    def make_prediction(
        self,
        data: Dataset,
        *args,
        proba: bool = False,
        threshold: float = 0.5,
        use_index: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Makes a prediction given an input. For example a customer number.
        Calls `load_prediction_data(*args)` and passes resulting data to `predict()`
        on the estimator

        Parameters
        ----------
        data: Dataset
            an instantiated Dataset object

        proba: bool
            Whether prediction is returned as a probability or not.
            Note that the return value is an n-dimensional array where n = number of classes

        use_index: bool
            Whether the index from the prediction data should be used for the result.

        use_cache: bool
            Whether or not to use the cached data in dataset to make predictions.
            Useful for seeing probability distributions of the model

        Returns
        -------
        pd.DataFrame
            A DataFrame with a prediction per row.
        """
        if proba is True and not hasattr(self.estimator, "predict_proba"):
            raise MLToolingError(
                f"{self.estimator_name} does not have a `predict_proba` method"
            )

        x = data.x if use_cache else data._load_prediction_data(*args, **kwargs)

        try:
            if proba:
                data = self.estimator.predict_proba(x)
                columns = [
                    f"Probability Class {col}" for col in self.estimator.classes_
                ]
            else:
                data = (
                    _classify(x, self.estimator, threshold=threshold)
                    if self.is_classifier
                    else self.estimator.predict(x)
                )
                columns = ["Prediction"]
            if use_index:
                prediction = pd.DataFrame(data=data, index=x.index, columns=columns)
            else:
                prediction = pd.DataFrame(data=data, columns=columns)

            return prediction

        except NotFittedError:
            message = (
                f"You haven't fitted the estimator. Call `.train_estimator` "
                f"or `.score_estimator` first"
            )
            raise MLToolingError(message) from None

    @classmethod
    def test_estimators(
        cls,
        data: Dataset,
        estimators: Sequence[Estimator],
        metrics: Union[str, List[str]] = "default",
        cv: Union[int, bool] = False,
        log_dir: str = None,
        refit: bool = False,
    ) -> Tuple["Model", ResultGroup]:
        """
        Trains each estimator passed and returns a sorted list of results

        Parameters
        ----------
        data: Dataset
            An instantiated Dataset object with train_test data

        estimators: Sequence[Estimator]
            List of estimators to train

        metrics: str, list of str
            Metric or list of metrics to use in scoring of estimators

        cv: int, bool
            Whether or not to use cross-validation. If an int is passed, use that many folds

        log_dir: str, optional
            Where to store logged estimators. If None, don't log

        refit: bool
            Whether or not to refit the best model on all the training data

        Returns
        -------
        List of Result objects
        """

        results = _train_estimators(estimators, data, metrics, cv)

        if log_dir:
            results.log(pathlib.Path(log_dir))

        best_estimator: Model = results[0].model
        logger.info(
            f"Best estimator: {best_estimator.estimator_name} - "
            f"{results[0].metrics.name}: {results[0].metrics.score}"
        )
        if refit:
            best_estimator.score_estimator(data)

        return best_estimator, results

    def train_estimator(self, data: Dataset) -> "Model":
        """
        Loads all training data and trains the estimator on all data.
        Typically used as the last step when estimator tuning is complete.

        .. warning::
            This will set self.result attribute to None. This method trains the estimator
            using all the data, so there is no validation data to measure results against

        Returns
        -------
        Model
            Returns an estimator trained on all the data, with no train-test split

        """
        logger.info("Training estimator...")
        self.estimator.fit(data.x, data.y)

        # Prevent confusion, as train_estimator does not return a result
        self.result = None
        logger.info("Estimator trained!")

        return self

    def score_estimator(
        self,
        data: Dataset,
        metrics: Union[str, List[str]] = "default",
        cv: Optional[int] = False,
    ) -> Result:
        """
        Scores the estimator based on training data from `data` and validates based on validation
        data from `data`.

        Defaults to no cross-validation. If you want to cross-validate the results,
        pass number of folds to cv. If cross-validation is used, `score_estimator` only
        cross-validates on training data and doesn't use the validation data.

        Returns a :class:`~ml_tooling.result.result.Result` object containing all result parameters

        Parameters
        ----------
        data: Dataset
            An instantiated Dataset object with create_train_test called

        metrics: string, list of strings
            Metric or metrics to use for scoring the estimator. Any sklearn metric string

        cv: int, optional
            Whether or not to use cross validation. Number of folds if an int is passed
            If False, don't use cross validation

        Returns
        -------
        Result
            A Result object that contains the results of the scoring
        """

        if isinstance(metrics, str):
            metrics = [self.default_metric] if metrics == "default" else [metrics]

        score_metrics = Metrics.from_list(metrics)
        cv = cv if cv else None

        if cv:
            cv = check_cv(cv, data.train_y, self.is_classifier)

        logger.info("Scoring estimator...")

        if not data.has_validation_set:
            raise DatasetError("Must run create_train_test first!")

        self.estimator.fit(data.train_x, data.train_y)

        if cv:
            logger.info("Cross-validating...")

        self.result = Result.from_model(
            model=self,
            data=data,
            metrics=score_metrics,
            cv=cv,
            n_jobs=self.config.N_JOBS,
            verbose=self.config.VERBOSITY,
        )

        if self.config.LOG:
            log = self.result.log(savedir=self.config.RUN_DIR)
            logger.info(f"Saved run info at {log.output_path}")
        return self.result

    def _cross_validated_search(
        self,
        data: Dataset,
        params: dict,
        estimators: Iterable[Estimator],
        metrics: Union[str, List[str]] = "default",
        cv: Optional[int] = None,
    ) -> Tuple["Model", ResultGroup]:
        """
        Runs a cross-validated search on the given estimators.

        Parameters
        ----------
        data: Dataset
            An instance of a DataSet object

        params: dict
            Parameters to use for search

        estimators: Iterable[Estimator]
            estimators to cross validate

        metrics: str, list of str
            Metrics to use for scoring. "default" sets metric equal to
            :attr:`self.default_metric`. First metric is used to sort results.

        cv: int, optional
            Cross validation to use. Defaults to value in :attr:`config.CROSS_VALIDATION`

        Returns
        -------
        best_estimator: Model
            Best estimator as found by the gridsearch

        result_group: ResultGroup
            ResultGroup object containing each individual score
        """
        if isinstance(metrics, str):
            metrics = self.default_metric if metrics == "default" else metrics

        cv = self.config.CROSS_VALIDATION if cv is None else cv
        cv = check_cv(cv, data.train_y, self.is_classifier)

        logger.debug(f"Cross-validating with {cv}-fold cv using {metrics}")
        logger.info("Starting search...")

        self.result = _train_estimators(
            list(estimators), data=data, metrics=metrics, cv=cv
        )

        logger.info("Done!")

        if self.config.LOG:
            result_file = self.result.log(self.config.RUN_DIR)
            logger.info(f"Saved run info at {result_file}")

        return self.result[0].model, self.result

    def gridsearch(
        self,
        data: Dataset,
        param_grid: dict,
        metrics: Union[str, List[str]] = "default",
        cv: Optional[int] = None,
    ) -> Tuple["Model", ResultGroup]:
        """
        Runs a cross-validated gridsearch on the estimator with the passed in parameter grid.

        Parameters
        ----------
        data: Dataset
            An instance of a DataSet object

        param_grid: dict
            Parameters to use for grid search

        metrics: str, list of str
            Metrics to use for scoring. "default" sets metric equal to
            :attr:`self.default_metric`. First metric is used to sort results.

        cv: int, optional
            Cross validation to use. Defaults to value in :attr:`config.CROSS_VALIDATION`

        Returns
        -------
        best_estimator: Model
            Best estimator as found by the gridsearch

        result_group: ResultGroup
            ResultGroup object containing each individual score
        """

        estimators: Iterable[Estimator] = prepare_gridsearch_estimators(
            estimator=self.estimator, params=param_grid
        )

        logger.debug(f"Gridsearching using {param_grid}")

        return self._cross_validated_search(data, param_grid, estimators, metrics, cv)

    def randomsearch(
        self,
        data: Dataset,
        param_distributions: dict,
        metrics: Union[str, List[str]] = "default",
        cv: Optional[int] = None,
        n_iter: int = 10,
    ) -> Tuple["Model", ResultGroup]:
        """
        Runs a cross-validated randomsearch on the estimator with a randomized
        sampling of the passed parameter distributions

        Parameters
        ----------
        data: Dataset
            An instance of a DataSet object

        param_distributions: dict
            Parameter distributions to use for randomizing search

        metrics: str, list of str
            Metrics to use for scoring. "default" sets metric equal to
            :attr:`self.default_metric`. First metric is used to sort results.

        cv: int, optional
            Cross validation to use. Defaults to value in :attr:`config.CROSS_VALIDATION`

        n_iter:
            Number of parameter settings that are sampled.

        random_state:
            Pseudo random number generator state used for random uniform sampling.

        Returns
        -------
        best_estimator: Model
            Best estimator as found by the randomsearch

        result_group: ResultGroup
            ResultGroup object containing each individual score
        """

        estimators: Iterable[Estimator] = prepare_randomsearch_estimators(
            estimator=self.estimator,
            params=param_distributions,
            n_iter=n_iter,
            random_state=self.config.RANDOM_STATE,
        )

        logger.debug(f"Randomsearching using {param_distributions}")

        return self._cross_validated_search(
            data, param_distributions, estimators, metrics, cv
        )

    @contextmanager
    def log(self, run_directory: str):
        """:meth:`log` is a context manager that lets you turn on logging for any scoring methods
        that follow. You can pass a log_dir to specify a subdirectory to store the estimator in.
        The output is a yaml file recording estimator parameters, package version numbers,
        metrics and other useful information

        Parameters
        ----------
        run_directory: str
            Name of the folder to save the details in

        Example
        --------
        If we want to log an estimator run in the `score` folder we can write::

             with estimator.log('score'):
                estimator.score_estimator

        This will save the results of `estimator.score_estimator()` to `runs/score/`

        """
        old_dir = self.config.RUN_DIR
        self.config.LOG = True
        self.config.RUN_DIR = self.config.RUN_DIR.joinpath(run_directory)
        try:
            yield
        finally:
            self.config.LOG = False
            self.config.RUN_DIR = old_dir

    @classmethod
    def load_production_estimator(cls, module_name):
        file_name = "production_model.pkl"
        with import_path(module_name, file_name) as path:
            estimator = joblib.load(path)
        return cls(estimator)

    @classmethod
    def reset_config(cls):
        """
        Reset configuration to default
        """
        cls._config = DefaultConfig()

        return cls

    def __repr__(self):
        return f"<Model: {self.estimator_name}>"


def _train_estimators(
    estimators: Sequence[Estimator],
    data: Dataset,
    metrics: Union[str, List[str]],
    cv: Any,
) -> ResultGroup:
    """
    Sequentially train a series of models and create a ResultGroup of the results

    Parameters
    ----------
    estimators: Sequence[Estimator]
        A sequence of estimators to compare
    data: DataSet
        Dataset object containing the training data
    metrics: Union[str, List[str]]
        Which metric to use to score the estimators
    cv: Any
        If an int is passed, perform that many CV splits. Alternatively, pass a sklearn CV object
        to use that for CV.
        If False, do not perform cross-validation

    Returns
    -------
    ResultGroup
        The results of the scoring
    """

    results = []
    for i, estimator in enumerate(estimators, start=1):
        challenger_estimator = Model(estimator)
        logger.info(
            f"Training estimator {i}/{len(estimators)}: {challenger_estimator.estimator_name}"
        )
        result = challenger_estimator.score_estimator(data, metrics=metrics, cv=cv)
        results.append(result)

    return ResultGroup(results).sort()
