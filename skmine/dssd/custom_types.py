from enum import Enum
from typing import Any, Callable, Dict, Hashable
import pandas
from .subgroup import Subgroup


FuncQuality = Callable[[Subgroup], float]


FuncCover = Callable[[Subgroup], pandas.DataFrame]


ColumnShares = Dict[str, Dict[Hashable, Any]]


class ColumnType(Enum):
    NOMINAL = "nominal"
    BINARY = "binary"
    NUMERIC = "numeric"
    TIME_SERIE = "time_serie"
