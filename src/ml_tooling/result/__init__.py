from typing import Union

from .result import Result
from .result_group import ResultGroup

ResultType = Union[Result, ResultGroup]

__all__ = ["Result", "ResultGroup"]
