from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from pandas import DataFrame
from .subgroup import Subgroup
from .description import Description
from .table import Table
from .cond import Cond
from .custom_types import ColumnType, FuncCover, FuncQuality
from .utils import get_cut_points, get_cut_points_smart

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

        for val in get_cut_points_smart(values, self.num_cut_points[col]):
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


### basic refinement operator ###
def generate_numeric_conditions(attribute: str, lower_bound: float, upper_bound: float, num_cut_points: int, operators_list: List[str]) -> List[Cond]:
    """Generate conditions to discretize the specified numeric attributes using the numeric operators specified on the current instance

    Args:
        attribute (str): the attribute to discretize
        lower_bound (float): the lowest value for the attribute
        upper_bound (float): the highest value for the attribute
        bins_count (int): the number of bins to create
        operators_list (List[str]): operators used for numeric conditions

    Returns:
        List[Cond]
    """
    cut_points = get_cut_points(lower_bound, upper_bound, num_cut_points)
    # cut_points = np.arange(left_bound, right_bound, (right_bound - left_bound) / bins_count)[1:]
    # make conditions
    conds: List[Cond] = []
    # for (_, end) in intervals:
    for point in cut_points:
        for op in operators_list:
            conds.append(Cond(attribute, op, point))  # new cond
    return conds


def generate_conditions(df: DataFrame, columns: List[str], column_types: Dict[str, str], discretization: Dict[str, int], operators_list: List[str] = [">", "<"]) -> List[Cond]:
    """Compute and return a list of conditions on the specified descriptive columns

    Args:
        columns (List[str]): attributes for which to generate conditions
        discretization (Dict[str, int]): number of bins to create per numeric attribute during discretization

    Returns:
        List[Cond]
    """
    conditions: List[Cond] = []
    for column_name in columns:  # Discretise pour chaque attribut numerique
        if column_types[column_name] == ColumnType.BINARY:
            # generate equal true/false condition for current attribute
            conditions.append(Cond(column_name, "==", True))
            conditions.append(Cond(column_name, "==", False))
        elif column_types[column_name] == ColumnType.NOMINAL:
            # create eq and not eq conditions for the unique values of the current attribute
            for val in df[column_name].unique():
                conditions.append(Cond(column_name, "==", val))
                conditions.append(Cond(column_name, "!=", val))
        elif column_types[column_name] == ColumnType.NUMERIC:
            lb = min(df[column_name].values)
            rb = max(df[column_name].values)
            conditions += generate_numeric_conditions(column_name, lb, rb, discretization[column_name], operators_list)
    return conditions


class RefinementOperatorBasic(RefinementOperator):
    def __init__(self, dataset: Table, num_cut_points: Dict[str, int], min_cov: int, cover_func: FuncCover, quality_func: FuncQuality) -> None:
        super().__init__(dataset, num_cut_points, min_cov, cover_func, quality_func)
        self.conditions: List[Cond] = generate_conditions(self.dataset.df, self.dataset.column_types.keys(), self.dataset.column_types,self.num_cut_points, ["<", ">"])


    def refine_candidate(self, cand: Subgroup, cand_list: List[Subgroup]):
        for cond in self.conditions:
            if cond not in cand.description.conditions:  # in DSSD paper
                new_candidate: Subgroup = None
                if self.dataset.column_types[cond.attribute] == ColumnType.BINARY:
                    if not cand.description.is_attribute_used(cond.attribute):
                        # add a new candidate that has this condition as an additional condition
                        new_candidate = Subgroup(Description([*cand.description.conditions, cond], cand.description.op), parent=cand)
                
                elif self.dataset.column_types[cond.attribute] == ColumnType.NOMINAL:
                    # if there is no condition that says nominal_attribute = "val" yet then we can add the current condition as it does not create any absurdity
                    if not any(c.attribute == cond.attribute and c.op == "=" for c in cand.description.conditions):
                        # add the new candidate with this condition in its pattern
                        new_candidate = Subgroup(Description([*cand.description.conditions, cond], cand.description.op), parent=cand)

                elif self.dataset.column_types[cond.attribute] == ColumnType.NUMERIC:
                    # we add this right away cause we know that we can have multiple conditions on the same numeric attribute even if it introdues redundant conditions they will be pruned later during dominance pruning
                    new_candidate = Subgroup(Description([*cand.description.conditions, cond], cand.description.op), parent=cand)

                if new_candidate is not None:
                    # update the candidate cover
                    self.check_and_append_candidate(new_candidate, cand_list)

        return cand_list


_builders: Dict[str, Callable[..., RefinementOperator]] = None

if _builders is None: 
    _builders = {
        "basic": RefinementOperatorBasic,
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
