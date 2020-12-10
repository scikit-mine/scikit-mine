"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
"""

# Author: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from collections import defaultdict
from itertools import takewhile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sortedcontainers import SortedDict

from ..utils import _check_min_supp
from ..utils import filter_maximal
from ..bitmaps import Bitmap

from ..base import BaseMiner, DiscovererMixin


class LCM(BaseMiner, DiscovererMixin):
    """
    Linear time Closed item set Miner.

    LCM can be used as a generic purpose miner, yielding some patterns
    that will be later submitted to a custom acceptance criterion.

    It can also be used to simply discover the set of closed itemsets from
    a transactional dataset.

    Parameters
    ----------
    min_supp: int or float, default=0.2
        The minimum support for itemsets to be rendered in the output
        Either an int representing the absolute support, or a float for relative support

        Default to 0.2 (20%)

    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single item is attributed a job
        to discover potential itemsets, considering this item as a root in the search space.
        Processes are preffered over threads.

    References
    ----------
    .. [1]
        Takeaki Uno, Masashi Kiyomi, Hiroki Arimura
        "LCM ver. 2: Efficient mining algorithms for frequent/closed/maximal itemsets", 2004

    .. [2] Alexandre Termier
        "Pattern mining rock: more, faster, better"

    Examples
    --------

    >>> from skmine.itemsets import LCM
    >>> from skmine.datasets.fimi import fetch_chess
    >>> chess = fetch_chess()
    >>> lcm = LCM(min_supp=2000)
    >>> patterns = lcm.fit_discover(chess)      # doctest: +SKIP
    >>> patterns.head()                         # doctest: +SKIP
        itemset support
    0      (58)    3195
    1  (11, 58)    2128
    2  (15, 58)    2025
    3  (17, 58)    2499
    4  (21, 58)    2224
    >>> patterns[patterns.itemset.map(len) > 3]  # doctest: +SKIP
    """

    def __init__(self, *, min_supp=0.2, n_jobs=1, verbose=False):
        _check_min_supp(min_supp)
        self.min_supp = min_supp  # provided by user
        self._min_supp = _check_min_supp(self.min_supp)
        self.item_to_tids_ = SortedDict()
        self.n_transactions_ = 0
        self.ctr = 0
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, D, y=None):
        """
        fit LCM on the transactional database, by keeping records of singular items
        and their transaction ids.
        """
        self.n_transactions_ = 0  # reset for safety
        item_to_tids = defaultdict(Bitmap)
        for transaction in D:
            for item in transaction:
                item_to_tids[item].add(self.n_transactions_)
            self.n_transactions_ += 1

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions_

        low_supp_items = [k for k, v in item_to_tids.items() if len(v) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        self.item_to_tids_ = SortedDict(item_to_tids)
        return self

    def discover(self, return_tids=False, max_depth=100):
        """Return the set of closed itemsets, with respect to the minium support

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        return_tids: bool
            Either to return transaction ids along with itemset.
            Default to False, will return supports instead

        max_depth: int, default=100
            Maximum depth for exploration in the search space.
            A root node is considered of depth 1.
            This can avoid cumbersome computation.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  =================================
                itemset     a `tuple` of co-occured items
                support     frequence for this itemset
                ==========  =================================

            if `return_tids=True` then
                ==========  =================================
                itemset     a `tuple` of co-occured items
                tids        a bitmap tracking positions
                ==========  =================================

        Example
        -------
        >>> from skmine.itemsets import LCM
        >>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        >>> LCM(min_supp=2).fit_discover(D)
             itemset  support
        0     (2, 5)        3
        1  (2, 3, 5)        2
        >>> LCM(min_supp=2).fit_discover(D, return_tids=True)  # doctest: +SKIP
             itemset       tids
        0     (2, 5)  [0, 1, 2]
        1  (2, 3, 5)     [0, 1]
        """
        empty_df = pd.DataFrame(columns=["itemset", "tids"])

        # reverse order of support
        supp_sorted_items = sorted(
            self.item_to_tids_.items(), key=lambda e: len(e[1]), reverse=True
        )

        dfs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._explore_root)(item, tids, max_depth)
            for item, tids in supp_sorted_items
        )

        dfs.append(empty_df)  # make sure we have something to concat
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.loc[:, "support"] = df["tids"].map(len).astype(np.uint32)
            df.drop("tids", axis=1, inplace=True)
        return df

    def _explore_root(self, item, tids, max_depth):
        it = self._inner((frozenset(), tids), item, 1, max_depth)
        df = pd.DataFrame(data=it, columns=["itemset", "tids"])
        if self.verbose and not df.empty:
            print("LCM found {} new itemsets from item : {}".format(len(df), item))
        return df

    def _inner(self, p_tids, limit, depth=1, max_depth=100):
        if depth >= max_depth:
            return
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.item_to_tids_.items())
            if tids.issubset(ids)
            if item not in p
        )

        # items are in reverse order, so the first consumed is the max
        max_k = next(takewhile(lambda e: e >= limit, cp), None)

        if max_k and max_k == limit:
            p_prime = (
                p | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()
            # sorted items in ouput for better reproducibility
            yield tuple(sorted(p_prime)), tids

            candidates = self.item_to_tids_.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids_[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    # new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit, depth + 1, max_depth)


class LCMMax(LCM):
    """
    Linear time Closed item set Miner.

    Adapted to Maximal itemsets (or borders).
    A maximal itemset is an itemset with no frequent superset.

    Parameters
    ----------
    min_supp: int or float, default=0.2
        The minimum support for itemsets to be rendered in the output
        Either an int representing the absolute support, or a float for relative support

        Default to 0.2 (20%)
    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single item is attributed a job
        to discover potential itemsets, considering this item as a root in the search space.
        Processes are preffered over threads.

    See Also
    --------
    LCM
    """

    def _inner(self, p_tids, limit, depth=1, max_depth=100):
        if depth >= max_depth:
            return
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.item_to_tids_.items())
            if tids.issubset(ids)
            if item not in p
        )

        max_k = next(
            cp, None
        )  # items are in reverse order, so the first consumed is the max

        if max_k and max_k == limit:
            p_prime = (
                p | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()

            candidates = self.item_to_tids_.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]

            no_cand = True
            for new_limit in candidates:
                ids = self.item_to_tids_[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    no_cand = False
                    # get new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit, depth + 1, max_depth)

            # only if no child node. This is how we PRE-check for maximality
            if no_cand:
                yield tuple(sorted(p_prime)), tids

    def discover(self, return_tids=False, max_depth=100):
        patterns = super().discover(return_tids=return_tids, max_depth=max_depth)
        maximums = [tuple(sorted(x)) for x in filter_maximal(patterns["itemset"])]
        return patterns[patterns.itemset.isin(maximums)]

    setattr(discover, "__doc__", LCM.discover.__doc__.replace("closed", "maximal"))
