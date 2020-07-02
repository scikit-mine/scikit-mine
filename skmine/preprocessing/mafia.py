"""
MAFIA: A Maximal Frequent Itemset Algorithm for Transactional Databases
"""

from collections import defaultdict

import pandas as pd
from roaringbitmap import RoaringBitmap
from sortedcontainers import SortedDict, SortedSet

from ..base import BaseMiner, DiscovererMixin


def dfs(item_to_tids, min_supp, head, head_tids, mfi):
    """
    Depth First Traversal
    Parameters
    ----------
    item_to_tids: SortedDict
        a sorted dictionary, mapping each item to its transaction ids
        in a vertical format
    """
    p = item_to_tids.bisect_right(head[-1])
    tail = item_to_tids.keys()[p:]
    tail = filter(
        lambda e: head_tids.intersection_len(item_to_tids[e]) >= min_supp,
        tail,
    )

    node = None
    for cand in tail:
        node = head | {cand} # TODO : pulling sortedkeys should be faster
        tids = head_tids.intersection(item_to_tids[cand])
        dfs(item_to_tids, min_supp, node, tids, mfi)
        # TODO : else break

    if node is None:
        head = frozenset(head)
        if any((e > head for e in mfi.keys())): return
        mfi[head] = head_tids



class Mafia(BaseMiner, DiscovererMixin):
    def __init__(self, min_supp=.2):
        self.min_supp_ = min_supp
        self.item_to_tids_ = None
        self.mfi_ = None

    def _prefit(self, D):
        d = defaultdict(RoaringBitmap)
        for tid, t in enumerate(D):
            for e in t:
                d[e].add(tid)

        self.item_to_tids_ = SortedDict(
            {k: v for k, v in d.items() if len(v) >= self.min_supp_}
        )

        return self

    def fit(self, D, y=None):
        self._prefit(D)

        mfi = dict()

        for item, tids in self.item_to_tids_.items():
            head = SortedSet([item])
            dfs(self.item_to_tids_, self.min_supp_, head, tids, mfi)

        self.mfi_ = mfi

        return self

    def discover(self):
        s = pd.Series(self.mfi_)
        s.index = s.index.map(tuple)
        return s
