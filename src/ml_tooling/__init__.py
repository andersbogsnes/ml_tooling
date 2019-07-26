# flake8: noqa
from .baseclass import ModelData
import pkg_resources

__version__ = pkg_resources.get_distribution("ml_tooling").version
__all__ = ["ModelData", "__version__"]
