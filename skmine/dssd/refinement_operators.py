from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from .subgroup import Subgroup
from .table import Table
from .cond import Cond
from .custom_types import ColumnType, FuncCover, FuncQuality
from .utils import _get_cut_points_smart

class RefinementOperator(ABC):
    def __init__(self, dataset: Table, num_cut_points: Dict[str, int], min_cov: int, cover_func: FuncCover, quality_func: FuncQuality) -> None:
        super().__init__()
        self.dataset = dataset
        self.column_types = dataset.column_types
        self.num_cut_points = num_cut_points
        self.min_cov = min_cov
        self.cover_func = cover_func
        self.quality_func = quality_func


    def check_and_append_candidate(self, cand: Subgroup, cand_list: List[Subgroup]):
        sg = self.cover_func(cand)
        if self.min_cov <= len(sg.index) < len(cand.parent.cover):
        # if self.min_cov <= len(sg.index) < len(self.dataset.df):
            cand.cover = sg.index
            cand.quality = self.quality_func(cand)
            cand_list.append(cand)
        return cand_list


    @abstractmethod
    def refine_candidate(self, cand: Subgroup, cand_list: List[Subgroup]) -> List[Subgroup]:
        pass


### official refinement operator ###
class RefinementOperatorOfficial(RefinementOperator):
    def __init__(self, dataset: Table, num_cut_points: Dict[str, int], min_cov: int, cover_func: FuncCover, quality_func: FuncQuality) -> None:
        super().__init__(dataset, num_cut_points, min_cov, cover_func, quality_func)


    def refine_binary(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        if cand.description.is_attribute_used(col):
            return
        self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "==", True)), cand_list)
        self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "==", False)), cand_list)


    def refine_nominal(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        unique_values = self.dataset.unique_df[col]
        if cand.description.is_attribute_used(col):
            if not cand.description.has_equal_condition_on_attr(col):
                for val in unique_values:
                    self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "!=", val)), cand_list)
        else:
            for val in unique_values:
                self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "==", val)), cand_list)
                self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "!=", val)), cand_list)


    def refine_numerical(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        # the way it is implemented in the dssd official source code
        values = sorted(self.dataset.df.loc[cand.cover][col].values)

        # having values be taken from here is what makes the cow patterns reach higher quality
        # values = self.dataset.df[col].values

        for val in _get_cut_points_smart(values, self.num_cut_points[col]):
            self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "<", val)), cand_list)
            self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, ">", val)), cand_list)


    def refine_column(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        if self.dataset.column_types[col] == ColumnType.BINARY:
            self.refine_binary(cand, col, cand_list)
        elif self.dataset.column_types[col] == ColumnType.NOMINAL:
            self.refine_nominal(cand, col, cand_list)
        elif self.dataset.column_types[col] == ColumnType.NUMERIC:
            self.refine_numerical(cand, col, cand_list)


    def refine_candidate(self, cand: Subgroup, cand_list: List[Subgroup]):
        for col_name in self.dataset.column_types:
            self.refine_column(cand, col_name, cand_list)
        return cand_list


class RefinementOperatorOfficialImproved(RefinementOperatorOfficial):
    def __init__(self, dataset: Table, num_cut_points: Dict[str, int], min_cov: int, cover_func: FuncCover, quality_func: FuncQuality) -> None:
        super().__init__(dataset, num_cut_points, min_cov, cover_func, quality_func)


    def check_and_append_candidate(self, cand: Subgroup, cand_list: List[Subgroup]):
        sg = self.cover_func(cand)
        if self.min_cov <= len(sg.index) < len(cand.parent.cover):
            cand.cover = sg.index
            cand.quality = self.quality_func(cand)
            if cand.quality > cand.parent.quality:
                cand_list.append(cand)
        return cand_list


_builders: Dict[str, Callable[..., RefinementOperator]] = None

if _builders is None: 
    _builders = {
        "official": RefinementOperatorOfficial,
        "official_improved": RefinementOperatorOfficialImproved,
        "powerfull": None
    }

def register(operator_name: str, builder: Callable[..., RefinementOperator]):
    _builders[operator_name] = builder

def create(operator_name: str, extra_parameters: Dict[str, Any]) -> RefinementOperator:
    builder = _builders.get(operator_name)
    if not builder:
            raise ValueError(f"!!! UNSUPPORTED REFINEMENT OPERATOR ({operator_name}) !!!")
    return builder(**extra_parameters)
