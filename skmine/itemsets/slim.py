"""SLIM pattern discovery"""

# Authors: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from collections import Counter, defaultdict
from functools import lru_cache, reduce
from itertools import chain

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from ..base import BaseMiner, MDLOptimizer
from ..bitmaps import Bitmap
from ..utils import _check_D, supervised_to_unsupervised


def _to_vertical(D):
    res = defaultdict(Bitmap)
    for idx, transaction in enumerate(D):
        for e in transaction:
            res[e].add(idx)
    return dict(res)


def _log2(values):
    res_index = values.index if isinstance(values, pd.Series) else None
    res = np.zeros(len(values), dtype=np.float32)
    res[values != 0] = np.log2(values[values != 0]).astype(np.float32)
    return pd.Series(res, index=res_index)


def cover(sct: dict, itemsets: list):
    """
    cover a standard codetable sct given itemsets

    Parameters
    ----------
    sct: dict[object, Bitmap]
        a standard codetable, i.e the vertical representation of a dataset
    itemsets: list[frozenset]
        itemsets from a given codetable

    Notes
    -----
        sct is modified inplace
    """
    covers = dict()
    for iset in itemsets:
        it = [sct[i] for i in iset]
        usage = reduce(Bitmap.intersection, it).copy() if it else Bitmap()
        covers[iset] = usage
        for k in iset:
            sct[k] -= usage
    return covers


def generate_candidates_big(codetable, stack=set(), depth=None):
    """
    Generate candidates, but does not sort output by estimated gain

    The result is a python generator, not an in-memory list

    This results in slightly less accurate candidate generation,
    but avoids computing candidates that will never be evaluated,
    if coupled with an early stopping strategy.

    Parameters
    ----------
    codetable: SortedDict[frozenset, Bitmap]
        A codetable, sorted in Standard Candidate Order

    stack: set[frozenset], defaut=set()
        A stack of already seen itemsets, which will not be considered in output
        Note that this function updates the stack, passed as a reference

    See Also
    --------
    generate_candidates
    """
    assert isinstance(codetable, SortedDict)
    depth = depth or int(np.log2(len(codetable)) * 1e2)
    for idx, (X, X_usage) in enumerate(codetable.items()):
        Y = codetable.items()[idx + 1 : idx + 1 + depth]
        _best_usage = 0
        best_XY = None
        for y, y_usage in Y:
            XY = X.union(y)
            if XY in stack:
                continue
            stack.add(XY)
            inter_len = y_usage.intersection_len(X_usage)
            if inter_len > _best_usage:
                _best_usage = inter_len
                best_XY = XY

        if best_XY is not None:
            yield best_XY, _best_usage


def generate_candidates(codetable, stack=set()):
    """
    assumes codetable is sorted in Standard Candidate Order
    """
    return sorted(
        generate_candidates_big(codetable, stack=stack),
        key=lambda e: e[1],
        reverse=True,
    )


class SLIM(BaseMiner, MDLOptimizer):
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
    pruning: bool, default=True
        Either to activate pruning or not. Pruned itemsets may be useful at
        prediction time, so it is usually recommended to set it to False
        to build a classifier. The model will be less concise, but will lead
        to more accurate predictions on average.
    n_items: int, default=200
        Number of most frequent items to consider for mining.
        As SLIM is highly dependant from the set of symbols from which
        it refines its codetable,
        lowering this argument will significantly improve runtime.

        Note: The reconstruction is lossless from this set of items. If the input data
        has more than `n_items` items, then the reconstruction will be lossy w.r.t this
        input data.
    n_iter_no_change: int, default=100
        Number of candidate evaluation with no improvement to count before stopping optimization.


    Examples
    --------
    >>> from skmine.itemsets import SLIM
    >>> D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
    >>> SLIM().fit(D).codetable  # doctest: +SKIP
    (butter, tea)         [2]
    (milk, bananas)    [0, 1]
    (cookies)          [1, 2]
    dtype: object

    References
    ----------
    .. [1]
        Smets, K & Vreeken, J
        "Slim: Directly Mining Descriptive Patterns", 2012

    .. [2] Gandhi, M & Vreeken, J
        "Slimmer, outsmarting Slim", 2014
    """

    def __init__(
        self, *, pruning=True, n_items=200, n_iter_no_change=100,
    ):
        self.n_iter_no_change = n_iter_no_change
        self.n_items = n_items
        self.tol_ = None
        self.standard_codetable_ = None
        self.codetable_ = SortedDict()
        self.model_size_ = None  # L(CT|D)
        self.data_size_ = None  # L(D|CT)
        self.pruning = pruning

    def fit(self, D, y=None):  # pylint:disable = too-many-locals
        """fit SLIM on a transactional dataset

        This generate new candidate patterns and add those which improve compression,
        iteratibely refining ``self.codetable_``

        Parameters
        -------
        D: pd.DataFrame
            Transactional dataset, either as an iterable of iterables
            or encoded as tabular binary data
        """
        self._prefit(D, y=y)
        n_iter_no_change = 0
        seen_cands = set()

        tol = len(self.standard_codetable_) ** 2

        while n_iter_no_change < self.n_iter_no_change:
            candidates = self.generate_candidates(stack=seen_cands)
            for cand, _ in candidates:
                data_size, model_size, update_d, prune_set = self.evaluate(cand)
                diff = (self.model_size_ + self.data_size_) - (data_size + model_size)

                if diff > 0.01:  # underflow
                    self.codetable_.update(update_d)
                    if self.pruning:
                        self.codetable_, data_size, model_size = self._prune(
                            self.codetable_, prune_set, model_size, data_size
                        )

                    self.data_size_ = data_size
                    self.model_size_ = model_size

                if diff < tol:
                    n_iter_no_change += 1
                    if n_iter_no_change > self.n_iter_no_change:
                        break  # inner break

            if not candidates:  # if empty candidate generation
                n_iter_no_change += self.n_iter_no_change  # force while loop to break

        self.tol_ = tol
        return self

    def decision_function(self, D):
        """Compute covers on new data, and return code length

        This function function is named ``decision_function`` because code lengths
        represent the distance between a point and the current codetable.

        Setting ``pruning`` to False when creating the model
        is recommended to cover unseen data, and especially when building a classifier.

        Parameters
        ----------
        D: pd.DataFrame or np.ndarray
            new data to make predictions on, in tabular format

        Example
        -------
        >>> from skmine.itemsets import SLIM; import pandas as pd
        >>> def to_tabular(D): return pd.Series(D).str.join('|').str.get_dummies(sep="|")
        >>> D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
        >>> new_D = to_tabular([['cookies', 'butter']])
        >>> slim = SLIM().fit(to_tabular(D))
        >>> slim.decision_function(new_D)
        0   -1.321928
        dtype: float32
        """
        D = _check_D(D)
        codetable = pd.Series(self.codetable_)
        D_sct = {
            k: Bitmap(np.where(D[k])[0])
            for k in D.columns
            if k in self.standard_codetable_
        }
        isets = [frozenset(i & D_sct.keys()) for i in codetable.index]
        covers = cover(D_sct, isets)

        mat = np.zeros(shape=(len(D), len(covers)))
        for idx, tids in enumerate(covers.values()):
            mat[tids, idx] = 1
        mat = pd.DataFrame(mat, columns=covers.keys())

        code_lengths = codetable.map(len)
        ct_codes = code_lengths / code_lengths.sum()
        codes = (mat * ct_codes).sum(axis=1).astype(np.float32)
        # positive sign on log2 to return negative distance : sklearn]
        r = _log2(codes)
        r[r == 0] = -np.inf  # zeros would fool a `shortest code wins` strategy
        return r

    def generate_candidates(self, stack=None, thresh=1e3):
        """
        Generate candidates from the current codetable (SLIM is any-time)

        Note that `stack` is updated during the execution of this method.

        Parameters
        ----------
        stack: set[frozenset], default=None
            a stack of already-seen candidates to be excluded
        thresh: int, default=1_000
            if the size of the current codetable is higher than `thresh`,
            candidate are generated on-the-fly, and remain unsorted. If not,
            they are returned in a list, sorted by decreasing order of estimated gain

        Returns
        -------
        iterator[tuple(frozenset, Bitmap)]
        """
        ct = SortedDict(self._standard_candidate_order, self.codetable.items())
        # if big number of elements in codetable, just take a generator, do not sort output
        gen = generate_candidates if len(ct) < thresh else generate_candidates_big
        return gen(ct, stack=stack)

    def evaluate(self, candidate):
        """
        Evaluate ``candidate``, considering the current codetable and a dataset ``D``

        Parameters
        ----------
        candidate: frozenset
            a new candidate to be evaluated

        Returns
        -------
        (float, float, dict, set)
            updated (data size, model size, codetable)
            and finally the set of itemsets for which usage decreased
        """
        idx = self.codetable_.bisect(candidate)
        ct = list(self.codetable_)
        ct.insert(idx, candidate)
        D = {k: v.copy() for k, v in self.standard_codetable_.items()}
        CTc = cover(D, ct)

        updated, decreased = {candidate: CTc[candidate]}, set()
        for iset, usage in self.codetable_.items():  # TODO useless is size is too big
            if usage != CTc[iset]:
                updated[iset] = CTc[iset]
                if len(CTc[iset]) < len(usage):
                    decreased.add(iset)

        data_size, model_size = self._compute_sizes(CTc)  # TODO pruning in evaluate

        return data_size, model_size, updated, decreased

    def reconstruct(self):
        """reconstruct the original data from the current `self.codetable_`"""
        n_transactions = (
            max(map(Bitmap.max, filter(lambda e: e, self.codetable_.values()))) + 1
        )

        D = pd.Series([set()] * n_transactions)
        for itemset, tids in self.codetable_.items():
            D.iloc[list(tids)] = D.iloc[list(tids)].map(itemset.union)
        return D.map(sorted)

    @lru_cache(maxsize=1024)
    def get_support(self, *items):
        """
        Get support from an itemset

        Note
        ----
        Items in an itemset must be passed as positional arguments

        Unseen items will throw errors
        """
        a = items[-1]
        tids = self.standard_codetable_[a]
        if len(items) > 1:
            return tids & self.get_support(*items[:-1])
        return tids

    def _standard_cover_order(self, itemset):
        """
        Returns a tuple associated with an itemset,
        so that many itemsets can be sorted in Standard Cover Order
        """
        return (-len(itemset), -len(self.get_support(*itemset)), tuple(itemset))

    def _standard_candidate_order(self, itemset):
        return (-len(self.get_support(*itemset)), -len(itemset), tuple(itemset))

    def _prefit(self, D, y=None):
        if hasattr(D, "ndim") and D.ndim == 2:
            D = _check_D(D)
            if y is not None:
                D = supervised_to_unsupervised(D, y)  # SKLEARN_COMPAT
            item_to_tids = {k: Bitmap(np.where(D[k])[0]) for k in D.columns}
        else:
            item_to_tids = _to_vertical(D)
        sct = pd.Series(item_to_tids)
        usage = sct.map(len).astype(np.uint32)
        usage = usage.nlargest(self.n_items)
        sct = sct[usage.index]
        self.standard_codetable_ = sct

        ct_it = ((frozenset([e]), tids) for e, tids in sct.items())
        self.codetable_ = SortedDict(self._standard_cover_order, ct_it)

        codes = -_log2(usage / usage.sum())
        self._starting_codes = codes

        # L(code_ST(X)) = L(code_CT(X)), because CT=ST
        self.model_size_ = 2 * codes.sum()

        self.data_size_ = (codes * usage).sum()

        return self

    def _compute_sizes(self, codetable):
        """
        Compute sizes for both the data and the model

        .. math:: L(D|CT)
        .. math:: L(CT|D)

        Parameters
        ----------
        codetable : Mapping
            A series mapping itemsets to their usage tids

        Returns
        -------
        tuple(float, float)
            (data_size, model_size)
        """
        isets, usages = zip(
            *((_[0], len(_[1])) for _ in codetable.items() if len(_[1]) > 0)
        )
        usages = np.array(usages, dtype=np.uint32)
        codes = -_log2(usages / usages.sum())

        counts = Counter(chain(*isets))
        stand_codes_sum = sum(
            self._starting_codes[item] * ctr for item, ctr in counts.items()
        )

        model_size = stand_codes_sum + codes.sum()  # L(CTc|D) = L(X|ST) + L(X|CTc)
        data_size = (codes * usages).sum()
        return data_size, model_size

    def _prune(self, codetable, prune_set, model_size, data_size):
        """post prune a codetable considering itemsets for which usage has decreased

        Parameters
        ----------
        codetable: SortedDict
        prune_set: set
            itemsets in ``codetable`` for which usage has decreased
        model_size: float
            current model_size for ``codetable``
        data_size: float
            current data size when encoding ``D`` with ``codetable``

        Returns
        -------
        new_codetable, new_data_size, new_model_size: SortedDict, float, float
            a tuple containing the pruned codetable, and new model size and data size
            w.r.t this new codetable
        """
        prune_set = {k for k in prune_set if len(k) > 1}  # remove singletons
        while prune_set:
            cand = min(prune_set, key=lambda e: len(codetable[e]))
            prune_set.discard(cand)

            ct = list(codetable)
            ct.remove(cand)

            D = {
                k: v.copy() for k, v in self.standard_codetable_.items()
            }  # TODO avoid data copies
            CTp = cover(D, ct)
            decreased = {
                k for k, v in CTp.items() if len(k) > 1 and len(v) < len(codetable[k])
            }

            d_size, m_size = self._compute_sizes(CTp)

            if d_size + m_size < model_size + data_size:
                codetable.update(CTp)
                del codetable[cand]
                prune_set.update(decreased)
                data_size, model_size = d_size, m_size

        return codetable, data_size, model_size
