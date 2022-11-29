"""SLIM pattern discovery"""

# Authors: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from collections import Counter, defaultdict
from functools import lru_cache, reduce
from itertools import chain

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from pyroaring import BitMap as Bitmap

from ..base import BaseMiner, InteractiveMiner, MDLOptimizer
from ..utils import _check_D, supervised_to_unsupervised


def _to_vertical(D, stop_items=None, return_len=False):
    if stop_items is None:
        stop_items = set()
    res = defaultdict(Bitmap)
    idx = 0
    for idx, transaction in enumerate(D):
        for e in transaction:
            if e in stop_items:
                continue
            res[e].add(idx)
    if return_len:
        return dict(res), idx + 1
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

    Note
    ----
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
        Y = codetable.items()[idx + 1: idx + 1 + depth]
        _best_usage = 0
        best_XY = None
        for y, y_usage in Y:
            XY = X.union(y)
            if XY in stack:
                continue
            stack.add(XY)
            inter_len = y_usage.intersection_cardinality(X_usage)
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


class SLIM(BaseMiner, MDLOptimizer, InteractiveMiner):
    """SLIM: Directly Mining Descriptive Patterns

    SLIM looks for a compressed representation of transactional data.
    This compressed representation if a set of descriptive patterns,
    and can be used to:

    - provide a natively interpretable modeling of this data
    - make predictions on new data, using this condensed representation as an encoding scheme


    Parameters
    ----------
    k: int, default=50
        Number of non-singleton itemsets to mine.
        A singleton is an itemset containing a single item.
    pruning: bool, default=True
        Either to activate pruning or not. Pruned itemsets may be useful at
        prediction time, so it is usually recommended to set it to `False`
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
    tol: float, default=0.5
        Minimum compression gain (in bits) for a candidate to be accepted


    Examples
    --------
    >>> from skmine.itemsets import SLIM
    >>> D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
    >>> SLIM().fit(D).discover(singletons=True, usage_tids=True)
    (bananas, milk)    (0, 1)
    (butter, tea)         (2)
    (cookies,)         (1, 2)
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
            self, *, k=50, pruning=True, n_items=200, tol=0.5,
    ):
        self._starting_codes = None
        self.n_items = n_items
        self.tol = tol
        self.standard_codetable_ = None
        self.codetable_ = SortedDict()
        self.model_size_ = None  # L(CT|D)
        self.data_size_ = None  # L(D|CT)
        self.pruning = pruning
        self.k = k

    def fit(self, D, y=None):  # pylint:disable = too-many-locals
        """fit SLIM on a transactional dataset

        This generates new candidate patterns and add those which improve compression,
        iteratibely refining ``self.codetable_``

        Parameters
        ----------
        D: iterable of iterables or array-like
            Transactional dataset, either as an iterable of iterables
            or encoded as tabular binary data
        """
        self.prefit(D, y=y)
        seen_cands = set()
        k = 0

        while k < self.k:
            candidates = self.generate_candidates(stack=seen_cands)
            for cand, _ in candidates:
                data_size, model_size, usages = self.evaluate(cand)
                diff = (self.model_size_ + self.data_size_) - \
                    (data_size + model_size)

                if diff >= self.tol:
                    self.update(
                        usages=usages, data_size=data_size, model_size=model_size
                    )

                    k = sum(map(lambda iset: len(iset) > 1, self.codetable_))
                if k >= self.k:
                    break

            if not candidates:  # if empty candidate generation
                Warning(
                    f"could not find `{self.k}` itemsets, try with a lower `tol`")
                break

        return self

    def decision_function(self, D):
        """Compute covers on new data, and return code length

        This function is named ``decision_function`` because code lengths
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

        See Also
        --------
        cover
        discover
        """
        mat = self.cover(D)
        code_lengths = self.discover(singletons=True, usage_tids=False)
        ct_codes = code_lengths / code_lengths.sum()
        codes = (mat * ct_codes).sum(axis=1).astype(np.float32)
        # positive sign on log2 to return negative distance : sklearn]
        r = _log2(codes)
        r[r == 0] = -np.inf  # zeros would fool a `shortest code wins` strategy
        return r

    def generate_candidates(self, stack=set()):
        """
        Generate candidates from the current codetable (SLIM is any-time)

        Note that `stack` is updated during the execution of this method.

        Parameters
        ----------
        stack: set[frozenset], default=None
            a stack of already-seen candidates to be excluded

        Returns
        -------
        iterator[tuple(frozenset, Bitmap)]
        """
        ct = SortedDict(self._standard_candidate_order,
                        self.codetable_.items())
        return generate_candidates(ct, stack=stack)

    def evaluate(self, candidate):
        """
        Evaluate ``candidate``, considering the current codetable and a dataset ``D``

        Parameters
        ----------
        candidate: frozenset
            a new candidate to be evaluated

        Returns
        -------
        (float, float, dict)
            updated (data size, model size, codetable)
        """
        idx = self.codetable_.bisect(candidate)
        ct = list(self.codetable_)
        ct.insert(idx, candidate)
        D = {k: v.copy() for k, v in self.standard_codetable_.items()}
        CTc = cover(D, ct)

        decreased = set()
        for iset, usage in self.codetable_.items():  # TODO useless is size is too big
            if len(CTc[iset]) < len(usage):
                decreased.add(iset)

        data_size, model_size = self._compute_sizes(CTc)

        if self.pruning:
            CTc, data_size, model_size = self._prune(
                CTc, decreased, model_size, data_size
            )

        return data_size, model_size, CTc

    def update(self, candidate=None, model_size=None, data_size=None, usages=None):
        """
        Update the current codetable.

        If `candidate` is passed as None, `model_size`, `data_size` and `usages` will be used
        If `candidate` is not None, `model_size`, `data_size` and `usages`
        will be computed by calling `.evaluate`

        Parameters
        ----------
        candidate: frozenset, default=None
            candidate to be inserted

        model_size: float, default=None
            new model size (in bits) to be set

        data_size: float
            new data size (in bits) to be set

        usages: dict, default=None
            optional for usage outside this class
            eg. if one simply needs to include an itemset in the current codetable
            as in interactive data mining

        Raises
        ------
        AssertionError
        """
        assert not (candidate is None and usages is None)
        if usages is None:
            data_size, model_size, usages = self.evaluate(candidate)
        to_drop = {c for c in self.codetable_.keys() - usages.keys()
                   if len(c) > 1}
        self.codetable_.update(usages)
        for iset in to_drop:
            del self.codetable_[iset]

        self.data_size_ = data_size
        self.model_size_ = model_size

    def cover(self, D):
        """
        cover unseen data

        items never seen are dropped out


        Examples
        --------
        >>> from skmine.itemsets import SLIM
        >>> D = ["ABC", "AB", "BCD"]
        >>> s = SLIM().fit(D)
        >>> s.cover(["BC", "AB"])
           (A, B)   (B,)   (C,)
        0   False   True   True
        1    True  False  False

        Returns
        -------
        pd.DataFrame
        """
        if hasattr(D, "shape") and len(D.shape) == 2:  # tabular
            D = _check_D(D)
            D_sct = {
                k: Bitmap(np.where(D[k])[0])
                for k in D.columns
                if k in self.standard_codetable_
            }
        else:  # transactional
            D_sct = _to_vertical(D)

        isets = self.discover(singletons=True, usage_tids=False)
        isets = isets[isets.index.map(set(D_sct).issuperset)]
        covers = cover(D_sct, isets.index)

        mat = np.zeros(shape=(len(D), len(covers)), dtype=bool)
        for idx, tids in enumerate(covers.values()):
            mat[tids, idx] = True
        return pd.DataFrame(mat, columns=list(covers.keys()))

    def discover(self, singletons=False, usage_tids=False, drop_null_usage=True):
        """Get a user-friendly copy of the codetable

        Parameters
        ----------
        singletons: bool, default=False
            Either to include itemsets of length 1 in the result
        usage_tids: bool, default=False
            Either to return transaction ids for an itemset (usage) or its codelength
        drop_null_usage: bool, default=True
            Either to include itemset with no usage in the training data
            (i.e itemsets under cover of other itemsets)

        Example
        -------
        >>> from skmine.itemsets import SLIM
        >>> D = ["ABC", "AB", "BCD"]
        >>> SLIM().fit(D).discover(singletons=True, usage_tids=True, drop_null_usage=False)
        (A, B)    (0, 1)
        (B,)         (2)
        (A,)          ()
        (C,)      (0, 2)
        (D,)         (2)
        dtype: object

        Returns
        -------
        pd.Series
            codetable containing patterns and ids of transactions in which they are used
        """
        s = {
            tuple(sorted(iset)): tids.copy()
            for iset, tids in self.codetable_.items()
            if len(tids) >= drop_null_usage and len(iset) > (not singletons)
        }
        s = pd.Series(list(s.values()), index=list(s.keys()))
        if not usage_tids:
            s = s.map(len).astype(np.uint32)
        return s

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
        so that many itemsets can be sorted in Standard Cover Order :
        |X|↓ supp_D(X) ↓ lexicographically ↑
        """
        return -len(itemset), -len(self.get_support(*itemset)), tuple(itemset)

    def _standard_candidate_order(self, itemset):
        """
        Returns a tuple associated with an itemset,
        so that many itemsets can be sorted in Standard Candidate Order :
        supp_D(X) ↓ |X| ↓ lexicographically ↑
        """
        return -len(self.get_support(*itemset)), -len(itemset), tuple(itemset)

    def prefit(self, D, y=None):
        """
        Parameters
        ----------
        D: iterable of iterables or array-like
            Transactional dataset, either as an iterable of iterables
            or encoded as tabular binary data

        Note
        ----
        works in 3 steps

        1. ingest data `D`
        2. track bitmaps for the top `self.n_items` frequent items from `D`
        3. set `self.data_size_` and `self.model_size` given the standard codetable
        """
        if hasattr(D, "ndim") and D.ndim == 2:
            D = _check_D(D)
            if y is not None:
                D = supervised_to_unsupervised(D, y)  # SKLEARN_COMPAT
            item_to_tids = {k: Bitmap(np.where(D[k])[0]) for k in D.columns}
        else:
            # compute tids for each item in Roaring Bitmap
            item_to_tids = _to_vertical(D)
        sct = pd.Series(item_to_tids)  # sct for "standard code table"
        # The usage of an itemset X ∈ CT (Code Table) is the number of transactions t ∈ D which have X in their cover.
        # A cover(t) is the set of itemsets X ∈ CT used to encode a transaction t
        # The Standard Codetable ST is composed of singletons as itemsets : so in ST
        #  - cover(t) = all singletons in t
        #  - usage(X) = supp(X) (support is the number of transactions in X )
        usage = sct.map(len).astype(np.uint32)
        usage = usage.nlargest(self.n_items)
        sct = sct[usage.index]

        # Build Standard Codetable <class 'pandas.core.series.Series'> in usage descending order :
        #   [
        #       singleton , tids      # first singleton with max usage
        #       singleton , tids
        #       ...
        #   ]
        self.standard_codetable_ = sct

        # Convert Standard Codetable pandas.Series in list of (  frozenset({.}), RoaringBitmap({...})   ,      ...    )
        ct_it = ((frozenset([e]), tids) for e, tids in sct.items())

        # Sort Standard Codetable in standard_cover_order
        self.codetable_ = SortedDict(self._standard_cover_order, ct_it)

        # Compute the length of the optimal prefix code of itemset X :
        #     codes(X) =  L(X|CT)  = -log2(usage(X) / ∑_CT usage)
        codes = -_log2(usage / usage.sum())
        self._starting_codes = codes

        # The encoded size of the code table is then given by
        # L(CT | D) = ∑_{X∈CT & usage(X)!=0) L(X|ST) + L(X|CT)
        #
        # because CT=ST :
        # L(CT | D) = 2 * ∑_{X∈CT & usage(X)!=0) L(X|ST)
        #           = 2 * ∑_{X} L(X|ST)
        #                (as X ∈ ST : usage(X)!=0 because it exists at least one transaction where {X} is in)
        #           = 2 * codes.sum()
        self.model_size_ = 2 * codes.sum()

        # Compute the length of the encoding of the database D : L(D | CT )
        #
        # L(D | CT ) is the sum of the sizes of the encoded transactions L(t|CT) for t ∈ D.
        #
        #   The length of the encoding of the transaction L(t|CT) is simply the
        #   sum of the code lengths of the itemsets in its cover :
        #             L(t|CT) =  ∑_{X∈cover(t)} L(X | CT )
        #
        #   As CT=ST  :  X ∈ cover(t) = {X} in t =>
        #                            L(t|CT)  =  ∑_{ {X} in t } L(X|CT)
        #                            L(t|CT)  =  ∑_{ {X} in t } codes(X)
        #
        #  L(D | CT)   = sum( L(t|CT) for  t∈D  )
        #              = ∑_{ t ∈ D }  ∑_{     {X} ∈ t   } codes(X)
        #              = ∑_{{X}∈ I }  ∑_{t ∈ tids(X) } codes(X)   (if we change place of the sum)
        #              = ∑_{{X}∈ I} usage(X) * codes(X)
        #
        # example : for D =  ['ABC', 'AB', 'BCD']
        # L(D | CT) =   codes('A') + codes('B') + codes('C')
        #             + codes('A') + codes('B')
        #             +              codes('B') + codes('C')  + codes('D')
        #           = 2codes('A') + 3codes('B') +2codes('C') + 1codes('D')
        #           = usage('A')*codes('A') + usage('B')*codes('B') + ...
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
