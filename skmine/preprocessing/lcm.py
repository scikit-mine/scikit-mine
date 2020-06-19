"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
"""

# Author: Rémi Adon <remi.adon@gmail.com>
# License: BSD 3 clause

from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from roaringbitmap import RoaringBitmap
from sortedcontainers import SortedDict


def _check_min_supp(min_supp):
    if isinstance(min_supp, int):
        if min_supp < 1:
            raise ValueError('Minimum support must be strictly positive')
    elif isinstance(min_supp, float):
        if min_supp < 0 or min_supp > 1:
            raise ValueError('Minimum support must be between 0 and 1')
    else:
        raise TypeError('Mimimum support must be of type int or float')
    return min_supp

class LCM():
    """
    Linear time Closed item set Miner.

    LCM can be used as a preprocessing step, yielding some patterns
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

    >>> from skmine.preprocessing import LCM
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
        self.item_to_tids = None
        self.n_transactions = 0
        self.ctr = 0
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _fit(self, D):
        self.n_transactions = 0  # reset for safety
        item_to_tids = defaultdict(RoaringBitmap)
        for transaction in D:
            for item in transaction:
                item_to_tids[item].add(self.n_transactions)
            self.n_transactions += 1

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions

        low_supp_items = [k for k, v in item_to_tids.items() if len(v) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        self.item_to_tids = SortedDict(item_to_tids)
        return self

    def fit_discover(self, D, return_tids=False):
        """fit LCM on the transactional database, and return the set of
        closed itemsets in this database, with respect to the minium support

        Different from ``fit_transform``, see the `Returns` section below.

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        return_tids: bool
            Either to return transaction ids along with itemset.
            Default to False, will return supports instead

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
        >>> from skmine.preprocessing import LCM
        >>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        >>> LCM(min_supp=2).fit_discover(D)
             itemset  support
        0     (2, 5)        3
        1  (2, 3, 5)        2
        >>> LCM(min_supp=2).fit_discover(D, return_tids=True)
             itemset       tids
        0     (2, 5)  [0, 1, 2]
        1  (2, 3, 5)     [0, 1]
        """
        self._fit(D)

        empty_df = pd.DataFrame(columns=['itemset', 'tids'])

        # reverse order of support
        supp_sorted_items = sorted(self.item_to_tids.items(), key=lambda e: len(e[1]), reverse=True)

        dfs = Parallel(n_jobs=self.n_jobs, prefer='processes')(
            delayed(self._explore_item)(item, tids) for item, tids in supp_sorted_items
        )

        dfs.append(empty_df) # make sure we have something to concat
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.loc[:, 'support'] = df['tids'].map(len).astype(np.uint32)
            df.drop('tids', axis=1, inplace=True)
        return df

    def fit_transform(self, D):
        """fit LCM on the transactional database, and encode the frequencies
        of the resulting patterns in a tabular format.

        This makes LCM a possible preprocessing step, compatible with ``scikit-learn``

        Notes
        -----
        Cells in the result will contain frequencies of patterns. Note that this process
        is somehow similar to Term-Frequency encoding, but operates on co-occuring terms
        instead of singular terms.

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database. Items must be both hashable and comparable

        Returns
        -------
        pd.DataFrame
            A DataFrame with itemsets as columns, and transactions as rows

        Example
        -------
        >>> from skmine.preprocessing import LCM
        >>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        >>> lcm = LCM(min_supp=2)
        >>> lcm.fit_transform(D)                # doctest: +SKIP
            2  3  5  # columns are single items w.r.t to the minium support
        0   2  2  2  # (2, 3, 5) has length 3 but support of 2
        1   2  2  2  # (2, 3, 5) has length 3 but support of 2
        2   3  0  3  # (2, 5) has length 2 but support of 3
        """
        patterns = self.fit_discover(D, return_tids=True)
        tid_s = patterns.set_index('itemset').tids
        by_supp = tid_s.map(len).sort_values(ascending=False)
        patterns = tid_s.reindex(by_supp.index)

        shape = (self.n_transactions, len(self.item_to_tids))
        mat = np.zeros(shape, dtype=np.uint32)

        df = pd.DataFrame(mat, columns=self.item_to_tids.keys())
        for pattern, tids in tid_s.iteritems():
            df.loc[tids, pattern] = len(tids)  # fill with support
        return df


    def _explore_item(self, item, tids):
        it = self._inner(frozenset(), tids, item)
        df = pd.DataFrame(data=it, columns=['itemset', 'tids'])
        if self.verbose and not df.empty:
            print('LCM found {} new itemsets from item : {}'.format(len(df), item))
        return df

    def _inner(self, p, tids, limit):
        # project and reduce DB w.r.t P
        cp = (
            item for item, ids in reversed(self.item_to_tids.items())
            if tids.issubset(ids) if item not in p
        )

        max_k = next(cp, None)  # items are in reverse order, so the first consumed is the max

        if max_k and max_k == limit:
            p_prime = p | set(cp) | {max_k}  # max_k has been consumed when calling next()
            # sorted items in ouput for better reproducibility
            yield tuple(sorted(p_prime)), tids

            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[:candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    new_limit_tids = tids.intersection(ids)
                    yield from self._inner(p_prime, new_limit_tids, new_limit)


def filter_maximal(itemsets):
    itemsets = [set(e) for e in itemsets]
    for iset in itemsets:
        if any(map(lambda e: e > iset, itemsets)):
            continue
        yield tuple(sorted(iset))

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
    def _inner(self, p, tids, limit):
        # project and reduce DB w.r.t P
        cp = (
            item for item, ids in reversed(self.item_to_tids.items())
            if tids.issubset(ids) if item not in p
        )

        max_k = next(cp, None)  # items are in reverse order, so the first consumed is the max

        if max_k and max_k == limit:
            p_prime = p | set(cp) | {max_k}  # max_k has been consumed when calling next()

            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[:candidates.bisect_left(limit)]

            no_cand = True
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    no_cand = False
                    new_limit_tids = tids.intersection(ids)
                    yield from self._inner(p_prime, new_limit_tids, new_limit)

            if no_cand:  # only if no child node. This is how we PRE-check for maximality
                yield tuple(sorted(p_prime)), tids # sorted items in ouput for better reproducibility

    def fit_discover(self, D, return_tids=False):
        patterns = super().fit_discover(D, return_tids=return_tids)
        maximums = list(filter_maximal(patterns['itemset']))
        return patterns[patterns.itemset.isin(maximums)]
