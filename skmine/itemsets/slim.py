"""SLIM pattern discovery"""

# Authors: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from collections import defaultdict
from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd
from roaringbitmap import RoaringBitmap

from ..base import BaseMiner
from ..utils import lazydict


def make_codetable(D: pd.Series):
    """
    Applied on an original dataset this makes up a standard codetable
    """
    codetable = defaultdict(RoaringBitmap)
    for idx, transaction in enumerate(D):
        for item in transaction:
            codetable[item].add(idx)
    return pd.Series(codetable)

def cover_one(codetable, cand):
    """
    assumes codetable is already sorted in Standard Cover Order
    """
    cover = list()
    stack = set()
    pos = 0
    while len(stack) < len(cand) and pos < len(codetable):
        iset = codetable[pos]
        pos += 1
        if not iset.isdisjoint(stack):
            continue
        if iset.issubset(cand):
            cover.append(iset)  # TODO add index instead of element for performance
            stack |= iset
    return cover


def generate_candidates(codetable):
    """
    assumes codetable is sorted in Standard Cover Order
    """
    res = list()
    for idx, (X, X_usage) in enumerate(codetable.iteritems()):
        Y = codetable.iloc[idx + 1:]
        XY_usage = Y.apply(lambda e: e.intersection_len(X_usage)).astype(np.uint32)
        XY_usage = XY_usage[XY_usage != 0]
        if not XY_usage.empty:
            best_Y = XY_usage.idxmax()
            best_XY = best_Y.union(X)
            res.append(best_XY)
    return res


class SLIM(BaseMiner): # TODO : inherit MDLOptimizer
    """ SLIM : mining itemsets TODO
    """
    def __init__(self, *, n_iter_no_change=5):
        self.n_iter_no_change = n_iter_no_change
        self.standard_codetable = None
        self.codetable = None
        self.supports = lazydict(self._get_support)
        self.model_size = None          # L(CT|D)
        self.data_size = None           # L(D|CT)

    def _get_support(self, itemset):
        U = reduce(RoaringBitmap.union, self.standard_codetable.loc[itemset])
        return len(U)

    def _get_cover_order_pos(self, codetable, cand):
        pos = 0
        while len(cand) < len(codetable[pos]):
            pos += 1
            if self.supports[cand] >= self.supports[codetable[pos - 1]]:
                break
            # TODO : add lexicographic order
        return pos

    def _prefit(self, D: pd.Series):
        self.standard_codetable = make_codetable(D)
        usage = self.standard_codetable.map(len).astype(np.uint32)

        sorted_index = sorted(usage.index, key=lambda e: (-usage[e], e))
        self.codetable = self.standard_codetable.reindex(sorted_index, copy=True)
        self.codetable.index = self.codetable.index.map(lambda e: frozenset([e]))

        codes = -np.log2(usage / usage.sum())
        self.model_size = 2 * codes.sum()      # L(code_ST(X)) = L(code_CT(X)), because CT=ST
        self.data_size = (codes * usage).sum()

        return self


    def get_standard_codes(self, index):
        """compute the size of a codetable index given the standard codetable"""
        flat_items = list(chain(*index))
        items, counts = np.unique(flat_items, return_counts=True)

        usages = self.standard_codetable.loc[items].map(len).astype(np.uint32)
        usages /= usages.sum()
        codes = -np.log2(usages)
        return codes * counts

    def compute_sizes(self, codetable):
        """
        Parameters
        ----------
        codetable : pd.Series
            A series mapping itemsets for their tids
            TODO : no need for tids here, only usages

        Returns
        -------
        tuple(int, int)
            (data_size, model_size)
        """
        usages = codetable.map(len).astype(np.uint32)
        codes = -np.log2(usages / usages.sum())

        stand_codes = self.get_standard_codes(codetable.index)

        model_size = stand_codes.sum() + codes.sum() # L(CTc|D) = L(X|ST) + L(X|CTc)
        data_size = (codes * usages).sum()
        return data_size, model_size

    def fit(self, D):
        self._prefit(D)
        CTc_index = None
        n_iter_no_change = 0
        is_better = True
        while n_iter_no_change < self.n_iter_no_change:
            candidates = generate_candidates(self.codetable)
            for cand in candidates:
                CT_index = self.codetable.index
                cand_pos = self._get_cover_order_pos(CT_index, cand)
                CTc_index = CT_index.insert(cand_pos, cand)

                covers = D.map(lambda t: cover_one(CTc_index, t))
                CTc = make_codetable(covers)
                data_size, model_size = self.compute_sizes(CTc)

                if data_size + model_size < self.data_size + model_size:
                    zeros_index = self.codetable.index.difference(CTc.index)
                    CTc = CTc.reindex(CTc_index)
                    for itemset in zeros_index:
                        CTc[itemset] = RoaringBitmap()
                    self.codetable = CTc
                    self.data_size = data_size
                    self.model_size = model_size
                else:
                    is_better = False
            if not is_better:
                n_iter_no_change += 1
