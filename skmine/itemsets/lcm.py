import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from roaringbitmap import RoaringBitmap
from sortedcontainers import SortedDict

from ..base import BaseMiner, TransformerMixin


def _check_min_supp(min_supp):
    if isinstance(min_supp, int):
        if min_supp < 1:
            raise ValueError('Minimum support must be strictly positive')
    elif isinstance(min_supp, float):
        if min_supp < 0 or min_supp > 1:
            raise ValueError('Minimum support must be between 0 and 1')
    else:
        raise TypeError('Mimimum support must be of type int or float')

class LCM(BaseMiner, TransformerMixin):
    def __init__(self, *, min_supp=2, n_jobs=1):
        _check_min_supp(min_supp)
        self.min_supp = min_supp  # provided by user
        self._min_supp = self.min_supp  # used by the algorithm, hence to be modified
        self.item_to_tids = SortedDict()
        self.n_transactions = 0
        self.ctr = 0
        self.n_jobs = n_jobs

    def fit(self, D):
        """fit LCM on the transactional database
        This simply iterates over transactions of D in order to keep
        track of every item and transactions ids related
        
        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable
        
        Returns
        -------
        self

        Examples
        --------
        >>> from skmine.datasets.fimi import load_chess
        >>> from skmine.itemsets import LCM
        >>> chess = load_chess()
        >>> lcm = LCM()
        >>> lcm.fit(chess)
        >>> print(list(lcm.item_to_tids))
        """
        for transaction in D:
            for item in transaction:
                if item in self.item_to_tids:
                    self.item_to_tids[item].add(self.n_transactions)
                else:
                    self.item_to_tids[item] = RoaringBitmap([self.n_transactions])
            self.n_transactions += 1

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp *= self.min_supp * self.n_transactions  

        return self

    partial_fit = fit

    def transform(self, D):
        items = filter(lambda e: len(e[1]) >= self._min_supp, self.item_to_tids.items())
        sorted_items = sorted(items, key=lambda e: len(e[1]), reverse=True)  # reverse order of support

        dfs = Parallel(n_jobs=self.n_jobs, prefer='processes')(
            delayed(self._explore_item)(item, tids) for item, tids in sorted_items
        )
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _explore_item(self, item, tids):
        it = self._inner(frozenset(), tids, item)
        df = pd.DataFrame(data=it, columns=['itemset', 'support'])
        df.support = df.support.astype(np.uint32)
        if not df.empty:
            print('LCM found {} new itemsets from item : {}'.format(len(df), item))
        return df


    def _inner(self, p, tids, limit):
        # project and reduce DB w.r.t P
        cp = (item for item, ids in reversed(self.item_to_tids.items()) if tids.issubset(ids) if item not in p)

        max_k = next(cp, None)  # items are in reverse order, so the first consumed is the max

        if max_k and max_k == limit:
            p_prime = p | set(cp) | {max_k}  # max_k has been consumed when calling next()
            yield p_prime, len(tids)

            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[:candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    new_limit_tids = tids.intersection(ids)
                    yield from self._inner(p_prime, new_limit_tids, new_limit)
