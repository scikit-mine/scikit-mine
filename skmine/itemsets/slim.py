"""SLIM pattern discovery"""

# Authors: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from functools import reduce, lru_cache
from itertools import chain

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from ..base import BaseMiner, MDLOptimizer
from ..bitmaps import Bitmap
from ..utils import supervised_to_unsupervised
from ..utils import _check_D
from ..callbacks import mdl_prints


def cover(itemsets: list, D: pd.DataFrame):
    """
    assert itemsets are sorted in Standard Cover Order
    D must be a pandas DataFrame containing boolean values
    """
    covers = list()
    mat = D.values

    _itemsets = list(map(D.columns.get_indexer, itemsets))

    for iset in _itemsets:
        parents = (v for k, v in zip(_itemsets, covers) if not set(iset).isdisjoint(k))
        rows_left = reduce(Bitmap.union, parents, Bitmap())
        rows_left.flip_range(0, len(D))
        _mat = mat[rows_left][:, iset]
        bools = _mat.all(axis=1)
        rows_where = np.where(bools)[0]
        rows_where += min(rows_left, default=0)  # pad indexes
        covers.append(Bitmap(rows_where))

    return pd.Series(covers, index=itemsets)


def cover_one(itemsets, cand):
    """
    assumes itemsets is already sorted in Standard Cover Order
    """
    cov = list()
    stack = set()
    for iset in itemsets:
        if not iset.isdisjoint(stack):
            continue
        if iset.issubset(cand):
            cov.append(iset)  # TODO add index instead of element for performance
            stack |= iset
        if len(stack) >= len(cand):
            break
    return cov


def generate_candidates_big(codetable, stack=None):
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

    stack: set[frozenset], defaut=None
        A stack of already seen itemsets, which will not be considered in output
        Note that this function updates the stack, passed as a reference

    See Also
    --------
    generate_candidates
    """
    for idx, (X, X_usage) in enumerate(codetable.items()):
        Y = codetable.items()[idx + 1 :]
        _best_usage = 0
        best_XY = None
        for y, y_usage in Y:
            XY = X.union(y)
            if stack is not None:
                if XY in stack:
                    continue
                stack.add(XY)
            inter_len = y_usage.intersection_len(X_usage)
            if inter_len > _best_usage:
                _best_usage = inter_len
                best_XY = X.union(y)

        if best_XY is not None:
            yield best_XY, _best_usage


def generate_candidates(codetable, stack=None):
    """
    assumes codetable is sorted in Standard Candidate Order
    """
    return sorted(
        generate_candidates_big(codetable, stack=stack),
        key=lambda e: e[1],
        reverse=True,
    )


def _update_usages(codetable: SortedDict, cand: frozenset, cand_usage: Bitmap):
    """
    Given a codetable and a new candidate:

    - iteratively consider every element in the codetable,
      starting from insertion position in Standard Cover Order
    - decrease usages when element is non disjoint with the candidate
    - identify usages that will increase due to freed intermediate covering,
      and increase them

    We also output the set of itemsets for which usage decreased, for later pruning.

    Notes
    -----
    In practice most usages will stay steady, but identifying those wich need to be
    modified makes updating the codetable easier in case of acceptance.

    On average this method is many orders of magnitude faster than covering the database
    entirely every time we need to evaluate a candidate.
    It's also very cheap in memory, as we only instantiate new Bitmaps for a restricted
    part of the codetable : the part that would need an update
    """
    update_d = dict()
    decreased = list()  # track decreasing non-singletons

    cand_pos = codetable.bisect(cand)

    for idx, iset in enumerate(codetable.islice(cand_pos, len(codetable))):
        if not iset.isdisjoint(cand):
            iset_tids = codetable[iset]
            if not codetable[iset].isdisjoint(cand_usage):
                update_d[iset] = iset_tids - cand_usage

                if len(iset) > 1:
                    decreased.append(iset)

                    to_cov = iset - cand
                    cov = cover_one(codetable.islice(idx + 1, len(codetable)), to_cov)
                    for e in cov:
                        update_d[e] = codetable[e].union(cand_usage)

    return update_d, decreased


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
    n_iter_no_change: int, default=100
        Number of candidate evaluation with no improvement to count before stopping optimization.
    tol: float, default=None
        Tolerance for the early stopping, in bits.
        When the compression size is not improving by at least tol for n_iter_no_change iterations,
        the training stops.
        Default to None, will be automatically computed considering the size of input data.
    pruning: bool, default=False
        Either to activate pruning or not. Pruned itemsets may be useful at
        prediction time, so it is usually recommended to set it to False
        to build a classifier. The model will be less concise, but will lead
        to more accurate predictions on average.
    verbose: integer
        Controls the verbosity: the higher, the more messages.


    Examples
    --------
    >>> from skmine.itemsets import SLIM
    >>> from skmine.preprocessing import TransactionEncoder
    >>> D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
    >>> D = TransactionEncoder().fit_transform(D)
    >>> SLIM().fit(D)                       # doctest: +SKIP
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

    def __init__(self, *, n_iter_no_change=100, tol=None, pruning=False, verbose=False):
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.standard_codetable_ = None
        self.codetable_ = pd.Series([], dtype="object")
        self.model_size_ = None  # L(CT|D)
        self.data_size_ = None  # L(D|CT)
        self.pruning = pruning
        self.verbose = verbose

        mdl_prints(self)  # attach mdl_prints <-- output if self.verbose set

    @lru_cache(maxsize=1024)
    def get_support(self, itemset):
        """Get support from an itemset"""
        U = reduce(Bitmap.union, self.standard_codetable_.loc[itemset])
        return len(U)

    def _standard_cover_order(self, itemset):
        """
        Returns a tuple associated with an itemset,
        so that many itemsets can be sorted in Standard Cover Order
        """
        # TODO : try returning a hash, sortedcontainers might prefer
        # handling integers when bisecting.
        return (-len(itemset), -self.get_support(itemset), tuple(itemset))

    def _standard_candidate_order(self, itemset):
        return (-self.get_support(itemset), -len(itemset), tuple(itemset))

    def _prefit(self, D):
        item_to_tids = {k: Bitmap(np.where(D[k])[0]) for k in D.columns}
        self.standard_codetable_ = pd.Series(item_to_tids)
        usage = self.standard_codetable_.map(len).astype(np.uint32)

        ct_it = ((frozenset([e]), tids) for e, tids in item_to_tids.items())
        self.codetable_ = SortedDict(self._standard_cover_order, ct_it)

        codes = -np.log2(usage / usage.sum())
        self.model_size_ = (
            2 * codes.sum()
        )  # L(code_ST(X)) = L(code_CT(X)), because CT=ST
        self.data_size_ = (codes * usage).sum()

        return self

    def generate_candidates(self, stack=None):
        ct = SortedDict(self._standard_candidate_order, self.codetable)
        # if big number of elements in codetable, just take a generator, do not sort output
        gen = generate_candidates if len(ct) < 1e3 else generate_candidates_big
        return gen(ct, stack=stack)

    def fit(self, D, y=None):  # pylint:disable = too-many-locals
        """fit SLIM on a transactional dataset

        This generate new candidate patterns and add those which improve compression,
        iteratibely refining ``self.codetable``
        """
        D = _check_D(D)
        if y is not None:
            D = supervised_to_unsupervised(D, y)  # SKLEARN_COMPAT

        self._prefit(D)
        n_iter_no_change = 0
        seen_cands = set()

        tol = self.tol or self.standard_codetable_.map(len).median()

        while n_iter_no_change < self.n_iter_no_change:
            candidates = self.generate_candidates(stack=seen_cands)
            for cand, _ in candidates:
                data_size, model_size, update_d, prune_set = self.evaluate(cand)
                diff = (self.model_size_ + self.data_size_) - (data_size + model_size)

                if diff > 0.01:  # underflow
                    self.codetable_.update(update_d)
                    if self.pruning:
                        self.codetable_, data_size, model_size = self._prune(
                            self.codetable_, D, prune_set, model_size, data_size
                        )

                    self.data_size_ = data_size
                    self.model_size_ = model_size

                if diff < tol:
                    n_iter_no_change += 1
                    if self.verbose:
                        print("n_iter_no_change : {}".format(n_iter_no_change))
                    if n_iter_no_change > self.n_iter_no_change:
                        break  # inner break

            if not candidates:  # if empty candidate generation
                n_iter_no_change += self.n_iter_no_change  # force while loop to break

        return self

    def decision_function(self, D):
        """Compute covers on new data, and return code length

        This function function is named ``decision_funciton`` because code lengths
        represent the distance between a point and the current codetable, i.e
        the probability for this point to belong to the codetable.

        The lower the better

        Setting ``pruning`` to False when creating the model
        is recommended to cover unseen data, and especially when building a classifier.

        Example
        -------
        >>> from skmine.preprocessing import TransactionEncoder
        >>> D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
        >>> te = TransactionEncoder()
        >>> D = te.fit_transform(D)
        >>> new_D = te.transform([['cookies', 'butter']])
        >>> slim = SLIM(pruning=False).fit(D)
        >>> slim.decision_function(new_D)
        0   -0.321928
        dtype: float32
        """
        D = _check_D(D)
        covers = cover(self.codetable.index, D)
        mat = np.zeros(shape=(len(D), len(covers)))
        for idx, tids in enumerate(covers.values):
            mat[tids, idx] = 1
        mat = pd.DataFrame(mat, columns=covers.index)

        ct_codes = self.codetable.map(len) / self.codetable.map(len).sum()
        codes = (mat * ct_codes).sum(axis=1)
        # positive sign on np.log2 to return negative distance : sklearn compat
        return np.log2(codes.astype(np.float32))

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
        ct = self.codetable_
        cand_pos = ct.bisect(candidate)

        # get original support from standard_codetable
        cand_usage = reduce(
            Bitmap.intersection, self.standard_codetable_.loc[candidate]
        )
        # remove union of all non disjoint itemsets before it to get real usage
        cand_usage -= reduce(
            Bitmap.union,
            (ct[k] for k in ct.islice(0, cand_pos) if not k.isdisjoint(candidate)),
            Bitmap(),
        )

        update_d, decreased = _update_usages(ct, candidate, cand_usage)

        update_d[candidate] = cand_usage
        CTc = {**ct, **update_d}

        data_size, model_size = self.compute_sizes(CTc)

        return data_size, model_size, update_d, decreased

    def get_standard_codes(self, index):
        """compute the size of a codetable index given the standard codetable"""
        flat_items = list(chain(*index))
        items, counts = np.unique(flat_items, return_counts=True)

        usages = self.standard_codetable_.loc[items].map(len).astype(np.uint32)
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
        codes = -np.log2(usages / usages.sum())

        stand_codes = self.get_standard_codes(isets)

        model_size = stand_codes.sum() + codes.sum()  # L(CTc|D) = L(X|ST) + L(X|CTc)
        data_size = (codes * usages).sum()
        return data_size, model_size

    def _prune(
        self, codetable, D, prune_set, model_size, data_size
    ):  # pylint: disable= too-many-arguments
        """post prune a codetable considering itemsets for which usage has decreased

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
        new_codetable, new_data_size, new_model_size: SortedDict, float, float
            a tuple containing the pruned codetable, and new model size and data size
            w.r.t this new codetable
        """
        prune_set = set(prune_set)
        while prune_set:
            cand = min(prune_set, key=lambda e: len(codetable[e]))
            prune_set.discard(cand)

            CTp_index = codetable.keys() - {cand}
            CTp_index = sorted(CTp_index, key=codetable.bisect)  # FIXME

            CTp = cover(CTp_index, D).to_dict()

            d_size, m_size = self.compute_sizes(CTp)

            if d_size + m_size < model_size + data_size:
                CTp = SortedDict(self._standard_cover_order, CTp)
                decreased = [k for k in CTp if len(CTp[k]) < len(codetable[k])]
                prune_set.update(decreased)
                codetable = CTp
                data_size, model_size = d_size, m_size

        return codetable, data_size, model_size
