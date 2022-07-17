from typing import Any, Callable, Dict, Hashable
import pandas
from .subgroup import Subgroup


FuncQuality = Callable[[Subgroup], float]


FuncCover = Callable[[Subgroup], pandas.DataFrame]


ColumnShares = Dict[str, Dict[Hashable, Any]]
