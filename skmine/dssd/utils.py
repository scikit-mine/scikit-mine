from collections import defaultdict
from typing import Collection, Dict, List, Type, Union
import numpy as np
from pandas import DataFrame
from pyparsing import Diagnostics

from .custom_types import ColumnShares, FuncQuality
from .subgroup import Subgroup
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


def sort_subgroups(subgroups: List[Subgroup], descending: bool = True):
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


def remove_duplicates(l: list):
    """Return a new list containing the unique elements of the specified list"""
    return list(set(l))


func_get_quality: FuncQuality = lambda c: c.quality


def column_shares(df: DataFrame, columns: List[str] = None) -> ColumnShares:
    """
    Compute and return a default dictionnary containing the shares/percentage for 
    each unique value for the specified columns in the dataframe

    Parameters
    ----------
    df: DataFrame
        The dataframe to extract unique values from
    columns: List[str], default=None
        The columns to compute shares for. All columns of the dataframe are used if this parameter is not specified. Defaults to None.

    Returns
    -------
    ColumnShares: a nested default dictionnary containing the shares

    Examples
    --------
    >>> from skmine.dssd.utils import column_shares
    >>> import pandas
    >>> df = pandas.DataFrame({"a": ["train", "train", "test", "train"], "b": [True, False, False, True]})
    >>> column_shares(df, ["a", "b"]) == {"a": {"train": 0.75, "test": 0.25 }, "b": {True: 0.5, False: 0.5}}
    True
    """
    columns = columns if columns is not None else df.columns
    return defaultdict(lambda: defaultdict(int, {}), {
        col: defaultdict(int, {
            val: val_count / len(df)
                for (val, val_count) in df[col].value_counts().items()}) 
                    for col in columns
    })


def subgroup(base_df: DataFrame, description: Description, only_check_last_cond: bool = False) -> DataFrame:
    """
    Return a pandas dataframe of elements that match the specified description

    Parameters
    ----------
    base_df: DataFrame
        The base dataframe from which selection is performed
    description: Description
        The description containing the conditions to be evaluated on the dataframe
    only_check_last_conditin: bool, default=False
        Whether or not to chech only the last condition in the description.
        This is particulary useful when one is sure that from the base dataframe
        is a direct parent of the subgroup which description is received as argument.
        That way as we know only one condition was added from the base_df to the
        current description, only that last condition should be evaluated

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> from skmine.dssd import subgroup, Cond
    >>> import pandas
    >>> df = pandas.DataFrame({"a": [2, 3, 4, 5], "bin": [1, 0, 0, 1], "cat": ["t", "t", "T", "T"]})
    >>> subgroup(df, Description([Cond("a", ">", 3), Cond("a", "<", 10), Cond("bin", "==", True), Cond("cat", "==", "T")] ))
       a  bin cat
    3  5    1   T
    """
    if len(description) == 0:
        return base_df
    
    if only_check_last_cond and len(description) > 1:
        description = Description([description.conditions[-1]])

    return base_df.query(str(description))


def min_max_avg(col: Collection[Union[int, float]]):
    """Return a tuple of three values:  minimum, maximum and average for the specified collection"""
    (_min,_max,_avg) = (None, None, None)
    count = 0
    for i in col:
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


def _min_max_avg_quality_string(cands: List[Subgroup], sep: str = "\n"):
    """Return a string version of the minimum, maximum and average quality for the considered candidates list"""
    (min_quality, max_quality, avg_quality) = min_max_avg(c.quality for c in cands)
    return f"min_quality={min_quality}{sep}max_quality={max_quality}{sep}avg_quality={avg_quality}"


def subgroups_to_csv(subgroups: List[Subgroup]) -> str:
    """Return a csv(like) representation of the specified subgroups"""
    return ("index,quality,size,#conditions,description\n" + \
        "\n".join(
        f"{index + 1},{sg.quality},{len(sg.cover)},{len(sg.description)},{sg.description}"
            for (index, sg) in enumerate(subgroups)
    )).replace(r"'", "")
