import numpy as np
import pandas as pd

from ..preprocessing.lcm import LCMMax, filter_maximal
from ..base import BaseMiner
from ..base import DiscovererMixin
from ..utils import _check_growth_rate

from itertools import combinations


def filter_minimal(itemsets):
    itemsets = list(itemsets)
    for iset in itemsets:
        if any(map(lambda e: e < iset, itemsets)):
            continue
        yield iset

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
    #assert len(R1) < len(U)  # assert we iterate on the smallest ensemble
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
        if any((e.issuperset(iset) for e in isets1)): continue
        inter = [iset & e for e in isets1]
        R = filter_maximal(inter)
        R = [set(e) for e in R]

        diff = border_diff(iset, R)
        borders.append(diff)

    return borders

def borders_to_patterns(left, right, min_size=None):
    """
    Parameters
    ----------
    left: list of set
    right: set
    min_size: int
        only accepts patterns with greater or equal size

    Operates in a bread-first manner, outputting all
    valid patterns of a given size for each level.
    Bigger patterns first.
    """
    min_size = min_size or min(map(len, left))
    for size in range(len(right) - 1, min_size - 1, -1):  # bigger patterns first
        combs = combinations(right, size)
        for pat in combs:
            if any((e.issuperset(set(pat)) for e in left)):
                continue
            yield pat


class MBDLLBorder(BaseMiner, DiscovererMixin):
    def __init__(self, min_growth_rate=2, min_supp=.1):
        self.min_supp_ = min_supp
        self.min_growth_rate_ = _check_growth_rate(min_growth_rate)
        self.borders_ = None


    def fit(self, D, y):
        labels = np.unique(y)
        assert len(labels) == 2
        assert isinstance(D, pd.Series)  # TODO : accept tabular data

        D1, D2 = D[y == labels[0]], D[y == labels[1]]

        # TODO : replace LCMMax by some more efficient method
        right_border_d1 = LCMMax(min_supp=self.min_supp_).fit_discover(D1)
        right_border_d2 = LCMMax(min_supp=self.min_growth_rate_ * self.min_supp_).fit_discover(D2)

        right_border_d1 = right_border_d1.itemset.map(set).tolist()
        right_border_d2 = right_border_d2.itemset.map(set).tolist()

        self.borders_ = mbdllborder(right_border_d1, right_border_d2)

        return self

    def discover(self):
        # TODO : post processing on border is necessary
        _, discriminative_patterns = zip(*self.borders_)
        return pd.Series(discriminative_patterns)
