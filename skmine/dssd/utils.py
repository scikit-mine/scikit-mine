from collections import defaultdict
from typing import Any, Collection, List
import numpy as np
from pandas import DataFrame

from .custom_types import FuncQuality
from .subgroup import Subgroup
from .cond import Cond
from .description import Description


def _get_cut_points(lo: float, hi: float, num_cut_points: int) -> List[float]:
    """
    Return a list of cutpoints obtained after binning the interval [start, end]
    into subbins each of amplitude equals to (max - min) / (num_cut_points + 1)
    Explanation: get_cut_points(0, 8, 4)
        Each bin is of size (8 - 0) / 4 = 2
        so we have the following bins: (0, 2) (2, 4) (4, 6) (6, 8)
        and among the bounds of those bins only 2, 4, 6 are useful 
        as any condition on either 0 or 8 is useless cause selection entire interval or nothing

    Parameters
    ----------
    lo: float
        The lower bound of the interval
    hi: float
        The upper bound of the interval 
    num_cut_points: int
        The number of cut points that we want for the interval 

    Returns
    -------
    List[float]

    Examples
    --------
    >>> from skmine.dssd.utils import _get_cut_points
    >>> _get_cut_points(lo=0, hi=10, num_cut_points=4) 
    [2.0, 4.0, 6.0, 8.0]
    """

    if num_cut_points <= 0:
        raise ValueError("The bins_count argument has to greater than 0")
    result = list(np.arange(lo, hi, (hi - lo) / (num_cut_points + 1)))
    return result[1:]


def _get_cut_points_smart(values: List[float], num_cut_points: int):
    """
    Compute and return a number of cut points for the specified values.
    This method aims to generate cut points that are all useful 
    meaning they actually take into account the distribution of 
    the values.
    Explanation:
        First a bin_size = len(values) / num_cut_points is computed and
        for i in 1 to num_cut_points, the actual cut point is computed by
        doing (values[bin_size * i - 1] + values[bin_size * i]) / 2.
        This way all the cut points generated actually cut the values
        in a usefull manner as this method does not only use lower and 
        upper bound of the values

    Parameters
    ----------
    values: List[float]: 
        The values for which to create cutpoints. Values need to be sorted ascending and 
        may be modified during the function execution 
    num_cut_points: int 
        The number of cut points that we want for the values 

    Returns:
    List[float]

    Examples
    --------
    >>> from skmine.dssd.utils import _get_cut_points_smart
    >>> list(_get_cut_points_smart([1, 5, 6, 6, 8], 2)) # [(5 + 6) / 2, (6 + 8)/2]
    [5.5, 7.0]
    """
    
    if len(values) <= num_cut_points:
        bin_size = 1
        num_cut_points = len(values) - 1
    else:
        bin_size = int(len(values) / num_cut_points)

    if bin_size * num_cut_points == len(values):
        # extend the values by the largest in order to prevent index error when the split point reaches the last value
        values.append(values[-1])

    return ((values[bin_size * i - 1] + values[bin_size * i]) / 2 for i in range(1, num_cut_points + 1))


def diff_items_count(l1: Collection, l2: Collection) -> int:
    """
    Return the number of items in the larger collection but not in the shorter one
    NB: This method doesn't handle duplicate values in the collections

    Parameters
    ----------
    l1: Collection
        The first collection
    l2: Collection
        The other collection

    Returns
    -------
    int

    Examples
    --------
    >>> from skmine.dssd.utils import diff_items_count
    >>> l1 = [1, 5, 6, 2]
    >>> l2 = [5, 7, 4, 2, 3]
    >>> diff_items_count(l1, l2)
    3
    """

    # always have l1 be the shorter list
    if len(l2) < len(l1):
        (l1, l2) = (l2, l1)

    return sum(1 for i in l2 if i not in l1)


def sort_candidates(subgroups: List[Subgroup], descending: bool = True):
    """
    Sort subgroups inplace based on their quality and for convenience,
    returns the subgroups list as a result 


    Parameters
    ----------
    subgroups: List[Subgroup])
        The list of subgroup to sort
    descending: bool, default=True
        Whether to sort descending or not

    Returns
    -------
    List[Subgroup]
    """
    subgroups.sort(key = func_get_quality, reverse=descending)
    return subgroups


def remove_duplicates(elements: list):
    return list(set(elements))


func_get_quality: FuncQuality = lambda c: c.quality


def column_shares(df: DataFrame, columns: List[str] = None):
    """Compute and return a default dictionnary containing the shares/percentage for each unique value 
    for the specified columns in the dataframe

    Args:
        df (DataFrame): the dataframe to extract unique values from
        columns (List[str], optional): the columns to compute shares for. All columns of the dataframe are used if this parameter is not specified. Defaults to None.

    Returns:
        ColumnShares: a default dictionnary containing the shares.
        This function returns a default dictionnary in order to allow natural writing like
        d = defaultdict(...)
        d["maybe_cola"]["unique_value_1"] and still get an actual result(0 if the column or value is not actually present in the result)
    """
    columns = columns if columns is not None else df.columns
    return defaultdict(lambda: defaultdict(int, {}), {
        col: defaultdict(int, {val: val_count / len(df) 
            for (val, val_count) in df[col].value_counts().items()}) 
                for col in columns
    })


def pattern_to_string(conditions: List[Cond]):
    return " & ".join([str(c) for c in conditions])


def eval_pattern_conditions(base_df: DataFrame, conditions: List[Cond]):
    res = base_df # simply grab a reference to the data frame
    for cond in conditions:
        if cond.op == "<=":
            res = res[res[cond.attribute] <= cond.val]
        elif cond.op == ">=":
            res = res[res[cond.attribute] >= cond.val]
        elif cond.op == "<":
            res = res[res[cond.attribute] < cond.val]
        elif cond.op == ">":
            res = res[res[cond.attribute] > cond.val]
        elif cond.op == "==":
            res = res[res[cond.attribute] == cond.val]
        elif cond.op == "!=":
            res = res[res[cond.attribute] != cond.val]
    return res
    # return base_df.query(pattern_to_string(conditions))


def sub_dict(d: dict, keys: list):
    return {k: d[k] for k in keys}


def subgroup(base_df: DataFrame, description: Description, only_check_last_cond: bool = False) -> DataFrame:
    """Return a subgroup of elements that match the specified description"""
    conditions = description.conditions if not only_check_last_cond else [description.conditions[-1]]
    return eval_pattern_conditions(base_df, conditions)


def min_max_avg(ar: list):
    """Return a tuple of thress values for the minimum, maximum and average quality for the considered candidates"""
    (_min,_max,_avg) = (None, None, None)
    count = 0
    for i in ar:
        if count == 0:
            (_min,_max,_avg) = (i,i,i)
        if i < _min:
            _min = i
        if i > _max:
            _max = i
        _avg += i
        count += 1

    if count == 0:
        return (0, 0, 0)

    return (_min,_max,_avg/count)


def min_max_avg_quality_string(cands: List[Subgroup], sep: str = "\n"):
    """Return a string version of the minimum, maximum and average quality for the considered candidates list"""
    (min_quality, max_quality, avg_quality) = min_max_avg(c.quality for c in cands)
    return f"min_quality={min_quality}{sep}max_quality={max_quality}{sep}avg_quality={avg_quality}"
    
    
def to_csv(cands: List[Subgroup]) -> str:
    return ("index,quality,size,#conditions,description\n" + \
        "\n".join(
        f"{index + 1},{cand.quality},{len(cand.cover)},{len(cand.description.conditions)},{pattern_to_string(cand.description.conditions)}"
            for (index, cand) in enumerate(cands)
    )).replace(r"'", "")
