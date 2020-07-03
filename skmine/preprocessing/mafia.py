"""
MAFIA: A Maximal Frequent Itemset Algorithm for Transactional Databases
"""

from collections import defaultdict

import pandas as pd
from roaringbitmap import RoaringBitmap
from sortedcontainers import SortedDict, SortedSet

from ..base import BaseMiner, DiscovererMixin

import logging

logger = logging.getLogger(__name__)


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

    node = None
    for cand in tail:
        cand_tids = item_to_tids[cand]
        if head_tids.issubset(cand_tids):
            head.append(cand)
        elif head_tids.intersection_len(cand_tids) >= min_supp:
            node = head + [cand]
            tids = head_tids.intersection(item_to_tids[cand])
            dfs(item_to_tids, min_supp, node, tids, mfi)

        else: break

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
        tid = 0

        for t in D:
            for e in t:
                d[e].add(tid)
            tid += 1

        min_supp = self.min_supp_ * tid if isinstance(self.min_supp_, float) else self.min_supp_

        self.item_to_tids_ = SortedDict(
            {k: v for k, v in d.items() if len(v) >= min_supp}
        )

        logger.info(f'keeping track of {len(self.item_to_tids_)} items')

        return self

    def fit(self, D, y=None):
        self._prefit(D)

        mfi = dict()

        for item, tids in self.item_to_tids_.items():
            logger.info(f'eploring item {item}')
            dfs(self.item_to_tids_, self.min_supp_, [item], tids, mfi)

        self.mfi_ = mfi

        return self

    def discover(self):
        s = pd.Series(self.mfi_)
        s.index = s.index.map(tuple)
        return s
