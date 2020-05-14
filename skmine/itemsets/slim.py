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


def generate_candidates(codetable, stack=None):
    """
    assumes codetable is sorted in Standard Cover Order
    """
    res = list()
    for idx, (X, X_usage) in enumerate(codetable.iteritems()):
        Y = codetable.iloc[idx + 1:]
        XY_usage = Y.apply(lambda e: e.intersection_len(X_usage)).astype(np.uint32)
        XY_usage = XY_usage[XY_usage != 0]
        XY_usage.index = XY_usage.index.map(X.union)
        if stack is not None:
            XY_usage = XY_usage[~XY_usage.index.isin(stack)]
        if not XY_usage.empty:
            best_XY = XY_usage.idxmax()
            res.append(best_XY)
    return res

class SLIM(BaseMiner): # TODO : inherit MDLOptimizer
    """SLIM: Directly Mining Descriptive Patterns

    SLIM looks for a compressed representation of transactional data.
    This compressed representation if a set of descriptive patterns,
    and can be used to:
        - provide a natively interpretable modeling of this data
        - make predictions on new data, using this condensed representation as an encoding scheme

    Idea of early stopping is inspired from
    http://eda.mmci.uni-saarland.de/pres/ida14-slimmer-poster.pdf

    Parameters
    ----------
    n_iter_no_change: int, default=5
        Number of iteration to count before stopping optimization.
    pruning: bool, default=True
        Either to activate pruning or not. Pruned itemsets may be useful at
        prediction time, so it is usually recommended to set it to False
        to build a classifier. The model will be less concise, but will lead
        to more accurate predictions on average.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmine.itemsets import SLIM
    >>> D = pd.Series([
    >>>     ['bananas', 'milk'],
    >>>     ['milk', 'bananas', 'cookies'],
    >>>     ['cookies', 'butter', 'tea'],
    >>>     ['tea'],
    >>>     ['milk', 'bananas', 'tea'],
    >>> ])
    >>> SLIM().fit(D)
        (milk, cookies, bananas)       [1]
        (butter, cookies, tea)         [2]
        (milk, bananas)             [0, 4]
        (tea)                       [3, 4]
        dtype: object

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
        self.codetable = pd.Series([])
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

    def __repr__(self): return repr(self.get_codetable())

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

    def fit(self, D):
        """ fit SLIM on a transactional dataset

        This generate new candidate patterns and add those which improve compression,
        iteratibely refining the ``self.codetable``
        """
        self._prefit(D)
        n_iter_no_change = 0
        is_better = False

        while n_iter_no_change < self.n_iter_no_change:
            is_better = False
            candidates = generate_candidates(self.codetable, stack=self._seen_cands)
            for cand in candidates:
                CTc, data_size, model_size = self.evaluate(cand, D)
                if data_size + model_size < self.data_size + self.model_size:
                    if self.pruning:
                        prune_set = CTc.drop([cand])
                        prune_set = prune_set[prune_set.map(len) < self.codetable.map(len)]
                        prune_set = prune_set[prune_set.index.map(len) > 1]
                        CTc, data_size, model_size = self._prune(
                            CTc, D, prune_set, model_size, data_size
                        )

                    self.codetable = CTc
                    self.data_size = data_size
                    self.model_size = model_size

                    is_better = True

                self._seen_cands.add(cand)

            if not is_better:
                n_iter_no_change += 1

        return self

    def predict_proba(self, D):
        """make predictions on a new transactional data

        This encode transactions with the current codetable.

        Example
        -------
        >>> D = pd.Series([
        >>>     ['bananas', 'milk'],
        >>>     ['milk', 'bananas', 'cookies'],
        >>>     ['cookies', 'butter', 'tea'],
        >>>     ['tea'],
        >>>     ['milk', 'bananas', 'tea'],
        >>> ])
        >>> new_D = pd.Series([['cookies', 'butter']])
        >>> slim = SLIM(pruning=False).fit(D)
        >>> slim.predict_proba(new_D)
        0    0.333333
        dtype: float32
        """
        assert isinstance(D, pd.Series)

        codetable = self.codetable[self.codetable.map(len) > 0]
        seen_items = frozenset(self.standard_codetable.index)
        D = D.map(seen_items.intersection)  # remove never seen items
        covers = D.map(lambda t: cover_one(codetable.index, t))
        ct_codes = codetable.map(len) / codetable.map(len).sum()
        codes = covers.map(lambda c: sum((ct_codes[e] for e in c)))
        return codes.astype(np.float32)

    def evaluate(self, candidate, D):
        """
        Evaluate ``candidate``, considering the current codetable and a dataset ``D``

        Parameters
        ----------
        candidate: frozenset
            a new candidate to be evaluated
        D: pd.Series
            a transactional dataset

        Returns
        -------
        (pd.Series, float, float, bool)
            updated (codetable, data size, model size
            and finally a boolean stating if compression improved
        """
        cand_pos = self._get_cover_order_pos(self.codetable.index, candidate)
        CTc_index = self.codetable.index.insert(cand_pos, candidate)

        covers = D.map(lambda t: cover_one(CTc_index, t))
        CTc = make_codetable(covers)
        data_size, model_size = self.compute_sizes(CTc)

        zeros_index = self.codetable.index.difference(CTc.index)  #FIXME
        CTc = CTc.reindex(CTc_index)
        for itemset in zeros_index:
            CTc[itemset] = RoaringBitmap()

        return CTc, data_size, model_size


    def get_codetable(self):
        """
        Returns
        -------
        pd.Series
            codetable containing patterns and ids of transactions in which they are used
        """
        return self.codetable[self.codetable.map(len) > 0]  # FIXME : this should not be needed

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
        Compute sizes for both the data and the model

        .. math:: L(D|CT)
        .. math:: L(CT|D)

        Parameters
        ----------
        codetable : pd.Series
            A series mapping itemsets to their usage tids

        Returns
        -------
        tuple(float, float)
            (data_size, model_size)
        """
        #TODO : no need for tids here, only usages
        usages = codetable.map(len).astype(np.uint32)
        codes = -np.log2(usages / usages.sum())

        stand_codes = self.get_standard_codes(codetable.index)

        model_size = stand_codes.sum() + codes.sum() # L(CTc|D) = L(X|ST) + L(X|CTc)
        data_size = (codes * usages).sum()
        return data_size, model_size

    def _prune(self, codetable, D, prune_set, model_size, data_size): # pylint: disable= too-many-arguments
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
        while not prune_set.empty:
            cand = prune_set.map(len).idxmin()
            prune_set = prune_set.drop([cand])
            CTp_index = codetable.index.drop([cand])
            covers = D.map(lambda t: cover_one(CTp_index, t))
            CTp = make_codetable(covers)
            d_size, m_size = self.compute_sizes(CTp)

            if d_size + m_size < model_size + data_size:
                zero_index = codetable.index.difference(CTp.index)
                zero_index = zero_index[zero_index.map(len) == 1]
                decreased = CTp.map(len) < codetable[CTp.index].map(len)
                prune_set.update(CTp[decreased])

                _u = {k: RoaringBitmap() for k in zero_index}
                CTp = pd.Series({**CTp, **_u})  # merge series

                codetable = CTp
                data_size, model_size = d_size, m_size

        return codetable, data_size, model_size
