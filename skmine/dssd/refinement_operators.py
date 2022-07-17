from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

from pandas import DataFrame, Series
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from .subgroup import Subgroup
from .cond import Cond
from .custom_types import FuncCover, FuncQuality
from .utils import _get_cut_points_smart

BINARY = "binary"
NUMERIC = "numeric"
NOMINAL = "nominal"

class RefinementOperator(ABC):
    """
    An operator which is used to generate newer valid subgroups 
    with longer descriptions based on one subgroup
    """

    def __init__(self, df: DataFrame, cover_func: FuncCover, quality_func: FuncQuality) -> None:
        super().__init__()
        self.df = df
        self.cover_func = cover_func
        self.quality_func = quality_func


    @property
    def df(self): return self._df

    @df.setter
    def df(self, df: DataFrame):
        self._df = df
        column_types = self.column_types = _column_types(df)
        self.unique_df: dict[str, Series] = {col: df[col].dropna(inplace=False).unique() for col in column_types if (column_types[col] == NOMINAL or column_types[col] == BINARY)}

    @abstractmethod
    def refine_candidate(self, cand: Subgroup) -> List[Subgroup]:
        """
        Generate candidates off of the specified cand and add the valid ones to the cand_list

        Parameters
        ----------
        cand: Subgroup
            The base candidate subgroup to refine

        Returns
        -------
        List[Subgroup]:
            A list of refined subgroups
        """
        pass


def _column_types(df: DataFrame) -> Dict[str, str]:
    """Creates column types to be used by refinement operators off of the dtypes of the dataframe following these rules
    dtype: bool -> binary, numeric -> numeric, other -> nominal
    """
    res = {}
    for col_name, dtype in df.dtypes.items():
        res[col_name] = BINARY if is_bool_dtype(dtype) else NUMERIC if is_numeric_dtype(dtype) else NOMINAL # any(fn(dtype) for fn in [pandas.api.types.is_string_dtype, pandas.api.types.is_object_dtype, pandas.api.types.is_categorical_dtype]):
    return res

class RefinementOperatorImpl(RefinementOperator):
    """
    Refinement operator as described in the official DSSD paper.

    Given a subgroup G, generates all valid subgroup descriptions that extend G’s description 
    with one condition. We distinguish three types of description attributes, each with its own specifics.

    * Binary attribute {==} : The only allowed condition type is ==, and consequently only a single condition on any binary attribute can be part of a subgroup description.
    * Nominal attribute {==, !=} : Both == and != are allowed. For any nominal attribute, either a single == or multiple != conditions are allowed in a description.
    * Numeric attribute {<, >} : Both < and > are allowed. Due to the large cardinality of numeric data, generating all \
    possible conditions is infeasible. Thus, to prevent the search space from exploding, the values \
    of a numeric attribute that occur within a subgroup are binned into six equal-sized bins \
    and {<, >}-conditions are generated for the five cut points obtained this way. This ‘on-the-fly’ \
    discretisation, performed upon refinement of a subgroup, results in a much more fine-grained\
    binning than a priori discretisation. Multiple conditions on the same attribute are\
    allowed, even though this may lead to redundant conditions in a description (e.g.D x < 10 ∧ D x < 5).

    Parameters
    ----------
    df: DataFrame, default=DataFrame()
        Inner pandas dataframe
    cover_func: FuncCover, default=None
        A function to compute the cover of generated subgroup in order to be valid
    quality_func: FuncQuality, default=None
        A quality function to compute quality of the generated subgroups
    min_cov: int, default=2
        The minimum coverage newer valid subgroups should have 
    min_quality: float, default=0.0
        The minimum quality generated subgroups should have to be considered valid 
    num_cut_points: Dict[str, int], default=defaultdict(lambda: 5)
        The number of cut points desired to disretize every single numeric descriptive attribute in the dataset

    Attributes
    ----------
    unique_df: Dict[str, Series]
        Map of binary/nominal columns and their unique values. This is stored 
        to avoid computing those unique values everytime as the inner dataset
        is likely not going to be modified
        
    References
    ----------
    [1] Page 225 (6.2 Refining subgroups)
        Leeuwen, Matthijs & Knobbe, Arno. (2012). Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 25. 10.1007/s10618-012-0273-y.
    """

    def __init__(self, df: DataFrame = DataFrame(), cover_func: FuncCover = None, quality_func: FuncQuality = None, min_cov: int = 2, min_quality: float = 0, num_cut_points: Dict[str, int] = defaultdict(lambda : 5)) -> None:
        """
        Create a new official refinement operator.
        If going to be used with the dssd algorithm, just do RefinementOperatorOfficial() 
        with no arguments, those will be filled in later by the algorithm.

        """
        super().__init__(df, cover_func, quality_func)
        self.num_cut_points = num_cut_points
        self.min_cov = min_cov
        self.min_quality = min_quality


    def __str__(self) -> str:
        return f"ref_op_impl[min_cov={self.min_cov}-min_quality={self.min_quality}-cut_points={dict(self.num_cut_points)}]"


    def check_and_append_candidate(self, cand: Subgroup, cand_list: List[Subgroup]):
        """
        This function is the one deciding the validity of a subgroup before including it in the result

        Parameters
        ----------
        cand: Subgroup
            The candidate subgroup to be evaluated
        cand_list: List[Subgroup]
            The list where to append the candidate if valid

        Returns
        -------
        List[Subgroup]: 
            The same result list received in parameter just for convenience purposes
        """
        sg = self.cover_func(cand)
        if self.min_cov <= len(sg.index) < len(self.df):
            cand.cover = sg.index
            cand.quality = self.quality_func(cand)
            if cand.quality >= self.min_quality:
                cand_list.append(cand)
        return cand_list


    def _refine_binary(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        if cand.description.is_attribute_used(col):
            return
        self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "==", True)), cand_list)
        self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "==", False)), cand_list)


    def _refine_nominal(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        unique_values = self.unique_df[col]
        if cand.description.is_attribute_used(col):
            if not cand.description.has_equal_condition_on_attr(col):
                for val in unique_values:
                    self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "!=", val)), cand_list)
        else:
            for val in unique_values:
                self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "==", val)), cand_list)
                self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "!=", val)), cand_list)


    def _refine_numerical(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        values = sorted(self.df.loc[cand.cover][col].values)
        for val in _get_cut_points_smart(values, self.num_cut_points[col]):
            self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, "<", val)), cand_list)
            self.check_and_append_candidate(cand.child_with_new_condition(new_cond = Cond(col, ">", val)), cand_list)


    def _refine_column(self, cand: Subgroup, col: str, cand_list: List[Subgroup]):
        if self.column_types[col] == BINARY:
            self._refine_binary(cand, col, cand_list)
        elif self.column_types[col] == NOMINAL:
            self._refine_nominal(cand, col, cand_list)
        elif self.column_types[col] == NUMERIC:
            self._refine_numerical(cand, col, cand_list)


    def refine_candidate(self, cand: Subgroup):
        res: List[Subgroup] = []
        for col_name in self.column_types:
            self._refine_column(cand, col_name, res)
        return res


class RefinementOperatorOfficialImproved(RefinementOperatorImpl):
    def check_and_append_candidate(self, cand: Subgroup, cand_list: List[Subgroup]):
        sg = self.cover_func(cand)
        if self.min_cov <= len(sg.index) < len(cand.parent.cover):
            cand.cover = sg.index
            cand.quality = self.quality_func(cand)
            if cand.quality >= cand.parent.quality:
                cand_list.append(cand)
        return cand_list
