from collections import defaultdict
from typing import Any, Collection, List
import numpy as np
from pandas import DataFrame

from .custom_types import FuncQuality
from .subgroup import Subgroup
from .cond import Cond
from .description import Description


def get_cut_points(min: float, max: float, num_cut_points: int) -> List[float]:
    """Return a list of cutpoints obtained after binning the (start, end) 
        each of amplitude equals to (max - min) / (num_cut_points + 1)


    Example: get_cut_points(0, 8, 4)
        Each bin is of size 8 - 0 / 4 = 2
        so we have the following bins: (0, 2) (2, 4) (4, 6) (6, 8)
        and among the bounds of those bins only 2, 4, 6 are usefull 
        as later setting a condition on either 0 or 8 will result in 
        selecting the whole interval or nothing from the interval


    Returns:
        List[float]:
    """

    if num_cut_points <= 0:
        raise ValueError("The bins_count argument has to greater than 0")
    result = list(np.arange(min, max, (max - min) / (num_cut_points + 1)))
    return result[1:]


def get_cut_points_smart(values: List[float], num_cut_points: int):
    if len(values) <= num_cut_points:
        bin_size = 1
        num_cut_points = len(values) - 1
    else:
        bin_size = int(len(values) / num_cut_points)
    # print(f"{len(values)=}, {bin_size=}")
    if bin_size * num_cut_points == len(values):
        # extend the values by the largest in order to prevent index error when the split point reaches the last value
        # values.append(0)
        values.append(values[-1])

    return ((values[bin_size * i - 1] + values[bin_size * i]) / 2 for i in range(1, num_cut_points + 1))
    #                                                    vvv bin_size * 1 instead of one and that resulted somehow in subgroups with a better quality overall I guess very likely because the subgroups were likely bigger as some conditions were highly likely to be kindof redundant
    # return ((values[bin_size * i - 1] + values[bin_size * 1]) / 2 for i in range(1, num_cut_points + 1))


def diff_items_count(l1: Collection[Cond], l2: Collection[Cond]) -> int:
    """Return the number of items in the largest collection but not in the shorter one"""

    # always have l1 be the shorter list
    if len(l2) < len(l1):
        (l1, l2) = (l2, l1)

    return sum(1 for i in l2 if i not in l1)


def sort_candidates(candidates: List[Subgroup], descending: bool = True):
    candidates.sort(key = lambda c: c.quality, reverse=descending)
    return candidates


def is_number_type(o: Any):
    return isinstance(o, (int, float))


def remove_keys_from_dict(d: dict, keys: Collection):
    for k in keys:
        del d[k]
    return d


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
