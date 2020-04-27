"""SLIM pattern discovery"""

# Authors: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
from roaringbitmap import RoaringBitmap

from ..base import BaseMiner, MDLOptimizer


class lazydict(defaultdict):
    """
    lazydict(default_factory[, ...]) --> dict with default factory

    The default factory is called with key as argument to produce
    a new value (via  __getitem__ only), and store it.
    A lazydict compares equal to a dict with the same items.
    All remaining arguments are treated the same as if they were
    passed to the dict constructor, including keyword arguments.
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        res = self[key] = self.default_factory(key)
        return res

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
    for iset in codetable:
        if not iset.isdisjoint(stack):
            continue
        if iset.issubset(cand):
            cover.append(iset)  # TODO add index instead of element for performance
            stack |= iset
        if len(stack) >= len(cand):
            break
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
        self.supports = lazydict(self.get_support)
        self.model_size = None          # L(CT|D)
        self.data_size = None           # L(D|CT)
    
    def get_support(self, itemset):
        U = reduce(RoaringBitmap.union, self.standard_codetable.loc[itemset])
        return len(U)

    def get_cover_order_pos(self, codetable, cand):
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
        self.codetable = self.standard_codetable.copy()
        self.codetable.index = self.codetable.index.map(lambda e: frozenset([e]))  # singletons
        self.codetable = self.codetable.reindex(usage.sort_values(ascending=False).index)

        codes = -np.log2(usage / usage.sum()) 
        self.model_size = 2 * codes.sum()      # L(code_ST(X)) = L(code_CT(X)), because CT=ST
        self.data_size = (codes * usage).sum()

        return self


    def fit(self): pass  # TODO