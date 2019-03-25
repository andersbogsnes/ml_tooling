# flake8: noqa
from .baseclass import ModelData
import pkg_resources

__all__ = ['ModelData']

__version__ = pkg_resources.get_distribution('ml_tooling').version
