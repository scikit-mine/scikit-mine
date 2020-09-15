"""
MBD-LLBorder
"""
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..base import BaseMiner, DiscovererMixin
from ..itemsets.lcm import LCMMax
from ..utils import _check_growth_rate, _check_min_supp, filter_maximal, filter_minimal


def border_diff(U, S):
    """
    Given a pair of borders <{∅}, {U}> and <{∅}, {S}>,
    ``border_diff`` derives another border <L2, {U}>
    such that [L2, {U}] = [{∅}, {U}] - [{∅}, {S}]

    Parameters
    ----------
    U : set
        non empty part from the border to differentiate

    S : list of set
        non-empty part of a border.
        Noted as ``R1`` in the original paper

    References
    ----------
    .. [1]
        Dong, Li
        Efficient Mining of Emerging Patterns Discovering

    Notes
    -----
    See ``BORDER-DIFF`` in section 4.1
    """
    # assert len(R1) < len(U)  # assert we iterate on the smallest ensemble
    L2 = [{x} for x in U - S[0]]
    for i in range(1, len(S)):
        _L2 = [X | {x} for x in U - S[i] for X in L2]
        L2 = list(filter_minimal(_L2))

    return L2, U


def mbdllborder(isets1, isets2):
    """
    References
    ----------
    .. [1]
        Dong, Li
        Efficient Mining of Emerging Patterns Discovering

    Notes
    -----
    main algorithm, as explained in section 4.2
    """
    borders = list()

    for iset in isets2:
        if any((e > iset for e in isets1)):
            continue
        inter = (iset & e for e in isets1)
        R = filter_maximal(inter)

        diff = border_diff(iset, R)
        borders.append(diff)

    return borders


def borders_to_patterns(left, right, min_size=None):
    """
    Operates in a bread-first manner, outputting all
    valid patterns of a given size for each level.
    Bigger patterns first.

    Parameters
    ----------
    left: list of set
    right: set
    min_size: int
        only accepts patterns with greater or equal size

    Returns
    -------
    pd.Series
    """
    min_size = min_size or min(map(len, left))
    patterns = list()
    for size in range(len(right) - 1, min_size - 1, -1):  # bigger patterns first
        combs = combinations(right, size)
        for pat in combs:
            if any((e.issuperset(set(pat)) for e in left)):
                continue
            patterns.append(pat)

    return pd.Series(patterns)


class MBDLLBorder(BaseMiner, DiscovererMixin):
    """
    MBD-LLBorder aims at discovering patterns characterizing the difference
    between two collections of data.

    It first discovers ``two sets of maximal itemsets``, one for each collection.
    It then looks for borders of these sets, and characterizes the difference
    by only manipulating theses borders.

    This results in the algorithm only keeping a ``concise description (borders)``
    as an internal state.

    Last but not least, it enumerates the set of emerging patterns from the
    borders.

    Parameters
    ----------
    min_growth_rate: int or float, default=2
        A pattern is considered as emerging iff its support in the first collection
        is at least min_growth_rate times its support in the second collection.

    min_supp: int or float, default=.1
        Minimum support in each of the collection
        Must be a relative support between 0 and 1
        Default to 0.1 (10%)

    n_jobs : int, default=1
        The number of jobs to use for the computation.


    Attributes
    ----------
    borders_: list of tuple[list[set], set]
        List of pairs representing left and right borders
        For every pair, emerging patterns will be uncovered by enumerating itemsets
        from the right side, while checking for non membership in the left side


    References
    ----------
    .. [1]
        Guozhu Dong, Jinyan Li
        "Efficient Mining of Emerging Patterns : Discovering Trends and Differences", 1999
    """

    def __init__(self, min_growth_rate=2, min_supp=0.1, n_jobs=1):
        self.min_supp = _check_min_supp(min_supp, accept_absolute=False)
        self.min_growth_rate = _check_growth_rate(min_growth_rate)
        self.borders_ = None
        self.n_jobs = n_jobs

    def fit(self, D, y):
        """
        fit MBD-LLBorder on D, splitted on y

        This is done in two steps
            1. Discover maximal itemsets for the two disinct collections contained in D
            2. Mine borders from these maximal itemsets

        Parameters
        ----------
        D : pd.Series
            The input transactional database
            Where every entry contains singular items
            Items must be both hashable and comparable

        y : array-like of shape (n_samples,)
            Targets on which to split D
            Must contain only two disctinct values, i.e len(np.unique(y)) == 2
        """
        labels = np.unique(y)
        assert len(labels) == 2
        assert isinstance(D, pd.Series)  # TODO : accept tabular data

        D1, D2 = D[y == labels[0]], D[y == labels[1]]

        # TODO : replace LCMMax by some more efficient method
        right_border_d1 = LCMMax(min_supp=self.min_supp).fit_discover(D1)
        right_border_d2 = LCMMax(
            min_supp=self.min_growth_rate * self.min_supp
        ).fit_discover(D2)

        right_border_d1 = right_border_d1.itemset.map(set).tolist()
        right_border_d2 = right_border_d2.itemset.map(set).tolist()

        self.borders_ = mbdllborder(right_border_d1, right_border_d2)

        return self

    def discover(self, min_size=3):
        """
        Enumerate emerging patterns from borders
        Subsets are drawn from the right borders, and are accepted iff
        they do not belong to the corresponding left border.

        This implementation is parallel, we consider every couple
        of right/left border in a separate worker and run the computation

        Parameters
        ----------
        min_size: int
            minimum size for an itemset to be valid
        """
        btp = delayed(partial(borders_to_patterns, min_size=min_size))
        series = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            btp(L, R) for L, R in self.borders_
        )
        return pd.concat(series) if series else pd.Series(dtype="object")
