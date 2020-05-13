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


def generate_candidates(codetable, stack=set()):
    """
    assumes codetable is sorted in Standard Cover Order
    """
    res = list()
    for idx, (X, X_usage) in enumerate(codetable.iteritems()):
        Y = codetable.iloc[idx + 1:]
        XY_usage = Y.apply(lambda e: e.intersection_len(X_usage)).astype(np.uint32)
        XY_usage = XY_usage[XY_usage != 0]
        XY_usage.index = XY_usage.index.map(X.union)
        XY_usage = XY_usage[~XY_usage.index.isin(stack)]
        if not XY_usage.empty:
            best_XY = XY_usage.idxmax()
            res.append(best_XY)
    return res

class SLIM(BaseMiner): # TODO : inherit MDLOptimizer
    """SLIM: Directly Mining Descriptive Patterns

    Idea of early stopping is inspired from
    http://eda.mmci.uni-saarland.de/pres/ida14-slimmer-poster.pdf

    Parameters
    ----------
    n_iter_no_change: int, default=5
        Number of iteration to count before stopping optimization.

    References
    ----------
    .. [1]
        Smets, K & Vreeken, J
        "Slim: Directly Mining Descriptive Patterns", 2012

    .. [2] Gandhi, M & Vreeken, J
        "Slimmer, outsmarting Slim", 2014
    """
    def __init__(self, *, n_iter_no_change=5, pruning=True):
        self.n_iter_no_change = n_iter_no_change
        self.standard_codetable = None
        self.codetable = None
        self.supports = lazydict(self._get_support)
        self.model_size = None          # L(CT|D)
        self.data_size = None           # L(D|CT)
        self.pruning = pruning
        self._seen_cands = set()  # set of previously seen items
        # TODO : add eps parameter for smarter early stopping

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

    def get_codetable(self):
        return self.codetable[self.codetable.map(len) > 0]  # FIXME : this should not be needed

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

    def prune(self, codetable, D, prune_set, model_size, data_size):
        """ post prune a codetable considering itemsets for which usage has decreased

        Parameters
        ----------
        codetable: pd.Series
        D: pd.Series
        prune_set: pd.Series
            subset of ``codetable`` for which usage has decreased
        models_size: float
            current model_size for ``codetable``
        data_size: float
            current data size when encoding ``D`` with ``codetable``

        Returns
        -------
        new_codetable, new_data_size, new_model_size: pd.Series, float, float
            a tuple containing the pruned codetable, and new model size and data size
            w.r.t this new codetable
        """
        new_model_size, new_data_size = model_size, data_size
        new_codetable = codetable  # TODO : remove would be enough

        while not prune_set.empty:
            cand = prune_set.map(len).idxmin()
            prune_set = prune_set.drop([cand])
            CTp_index = codetable.index.drop([cand])
            covers = D.map(lambda t: cover_one(CTp_index, t))
            CTp = make_codetable(covers)
            d_size, m_size = self.compute_sizes(CTp)

            if d_size + m_size < new_model_size + new_data_size:
                zero_index = new_codetable.index.difference(CTp.index)
                zero_singleton_index = zero_index[zero_index.map(len) == 1]
                decreased = CTp.map(len) < new_codetable[CTp.index].map(len)
                prune_set.update(CTp[decreased])

                _u = {k: RoaringBitmap() for k in zero_singleton_index}
                CTp = pd.Series({**CTp, **_u})  # merge series

                new_codetable = CTp
                new_data_size = d_size
                new_model_size = m_size

        return new_codetable, new_data_size, new_model_size


    def fit(self, D):
        """ fit SLIM on a transactional dataset

        This generate new candidate patterns and add those which improve compression,
        iteratibely refining the ``self.codetable``
        """
        self._prefit(D)
        CTc_index = None
        n_iter_no_change = 0
        while n_iter_no_change < self.n_iter_no_change:
            is_better = False
            candidates = generate_candidates(self.codetable, stack=self._seen_cands)
            for cand in candidates:
                CT_index = self.codetable.index
                cand_pos = self._get_cover_order_pos(CT_index, cand)
                CTc_index = CT_index.insert(cand_pos, cand)

                covers = D.map(lambda t: cover_one(CTc_index, t))
                CTc = make_codetable(covers)
                data_size, model_size = self.compute_sizes(CTc)

                if data_size + model_size < self.data_size + model_size:
                    zeros_index = self.codetable.index.difference(CTc.index)  #FIXME
                    CTc = CTc.reindex(CTc_index)
                    for itemset in zeros_index:
                        CTc[itemset] = RoaringBitmap()

                    if self.pruning:  # TODO : remove in the future, this is for testing purposes
                        prune_set = CTc.drop([cand])
                        prune_set = prune_set[prune_set.map(len) < self.codetable.map(len)]
                        prune_set = prune_set[prune_set.index.map(len) > 1]
                        CTc, data_size, model_size = self.prune(
                            CTc, D, prune_set, model_size, data_size
                        )

                    self.codetable = CTc
                    self.data_size = data_size
                    self.model_size = model_size
                    is_better = True
                self._seen_cands.add(cand)  # TODO : move to inner if statement ?

            if not is_better:
                n_iter_no_change += 1

        return self
