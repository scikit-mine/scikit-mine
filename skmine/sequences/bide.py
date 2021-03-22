"""
BIDE
"""

# Authors: Chuancong Gao
#          Cheng-Yuan Yu <yu8861213@hotmail.com>
#          Yuki Ueda <ikuyadeu0513@gmail.com>
#          Remi Adon <remi.adon@gmail.com>
#
# License: BSD 3 clause

from collections import defaultdict

import pandas as pd

from skmine.base import BaseMiner, DiscovererMixin
from skmine.utils import _check_min_supp


class BIDE(BaseMiner, DiscovererMixin):
    """
    Linear time Closed item set Miner.

    LCM can be used as a **generic purpose** miner, yielding some patterns
    that will be later submitted to a custom acceptance criterion.

    It can also be used to simply discover the set of **closed itemsets** from
    a transactional dataset.

    Parameters
    ----------
    min_supp: int or float, default=2
        The minimum support, i.e the minimum frequence for a pattern to be valid

    k: int, default=None
        Number of patterns to mine. Using this parameter,
        turns BIDE into a top-k closed sequential miner

    min_len: int, default=2
        minimum length for a pattern to be valid

    max_len: int, default=10
        maximum length for a pattern to be valid

    References
    ----------
    .. [1]
        Jianyong Wang, Jiawei Han.
        "BIDE: Efficient Mining of Frequent Closed Sequences.", 2004

    Examples
    --------
    >>> from skmine.sequences import BIDE
    >>> bide = BIDE()
    >>> db = [["wake up", "take shower", "make coffee"], ["wake up", "take shower", "make tea"], ["wake up", "take shower", "make tea", "start coding"]]
    >>> bide.fit_discover(db)
    (wake up, take shower)              3
    (wake up, take shower, make tea)    2
    dtype: int64
    """

    def __init__(self, min_supp=2, k=None, min_len=2, max_len=10):
        self.k = k
        self.min_supp = _check_min_supp(min_supp)
        self.min_len = min_len
        self.max_len = max_len
        self._db = None
        self._results = list()

    def fit(self, D):
        self._results.clear()
        self._db = D  # TODO this is dummy
        return self

    def discover(self):
        if self._db is None:
            raise Exception("must call fit before")
        matches = [(i, -1) for i in range(len(self._db))]
        self.bide_frequent_rec([], matches)
        patterns, supports = zip(*self._results)
        return pd.Series(supports, index=map(tuple, patterns))

    def __reversescan(self, db, patt, matches, check_type):
        # db: complete database
        # patt: the current pattern
        # matches: a list of tuples (row_index, the index of the last element of patt within db[row_index])
        def islocalclosed(previtem):
            closeditems = set()

            for k, (i, endpos) in enumerate(matches):
                localitems = set()

                for startpos in range(endpos - 1, -1, -1):
                    item = db[i][startpos]

                    if item == previtem:
                        matches[k] = (i, startpos)
                        break

                    localitems.add(item)

                # first run: add elements of localitems to closeditems
                # after first run: start intersection
                (closeditems.update if k == 0 else closeditems.intersection_update)(
                    localitems
                )

            return len(closeditems) > 0

        check = True if check_type == "closed" else False
        for previtem in reversed(patt[:-1]):

            if islocalclosed(previtem):
                check = False if check_type == "closed" else True
                break

        return check

    def isclosed(self, db, patt, matches):
        return self.__reversescan(
            db, [None, *patt, None], [(i, len(db[i])) for i, _ in matches], "closed"
        )

    def canclosedprune(self, db, patt, matches):
        return self.__reversescan(db, [None, *patt], matches[:], "prune")

    def bide_frequent_rec(self, patt, matches):

        sup = len(matches)

        # if pattern's length is greater than minimum length, consider whether it should be recorded
        if len(patt) >= self.min_len:

            # if pattern's support < minsup, stop
            if sup < self.min_supp:
                return None
            # if pattern is closed (backward extension check), record the pattern and its support
            if self.isclosed(self._db, patt, matches):
                self._results.append((patt, sup))

        # if pattern's length is greater than maximum length, stop recurssion
        if len(patt) == self.max_len:
            return None

        # find the following items
        occurs = nextentries(self._db, matches)
        for newitem, newmatches in occurs.items():
            # set the new pattern
            newpatt = patt + [newitem]

            # forward closed pattern checking
            if (len(matches) == len(newmatches)) and ((patt, sup) in self._results):
                self._results.remove((patt, sup))

            # can we stop pruning the new pattern
            if self.canclosedprune(self._db, newpatt, newmatches):
                continue
            self.bide_frequent_rec(newpatt, newmatches)


def invertedindex(seqs, entries=None) -> dict:
    index = defaultdict(list)

    for k, seq in enumerate(seqs):
        i, lastpos = entries[k] if entries else (k, -1)

        for p, item in enumerate(seq, start=(lastpos + 1)):
            l = index[item]
            if l and l[-1][0] == i:
                continue

            l.append((i, p))

    return index


def nextentries(data, entries):
    return invertedindex((data[i][lastpos + 1 :] for i, lastpos in entries), entries)
