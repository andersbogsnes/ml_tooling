# flake8: noqa
from .baseclass import BaseClassModel
import pkg_resources

__all__ = ['BaseClassModel']

__version__ = pkg_resources.get_distribution('pip').version
