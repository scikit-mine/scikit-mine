"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
"""

# Authors: Rémi Adon <remi.adon@gmail.com>
#          Luis Galárraga <galarraga@luisgalarraga.de>
#          Hermann Courteille <hermann.courteille@irisa.fr>
#          Cyril Regan <cyril.regan@loria.fr>
#          Thomas Betton <thomas.betton@irisa.fr>
#
# License: BSD 3 clause

from collections import defaultdict
from itertools import takewhile
import pandas as pd
import os
import shutil
from joblib import Parallel, delayed
from sortedcontainers import SortedDict
from pyroaring import BitMap as Bitmap

from ..utils import _check_min_supp
from ..utils import filter_maximal

from ..base import BaseMiner


class LCM(BaseMiner):
    """
    Linear time Closed item set Miner.

    LCM can be used as a **generic purpose** miner, yielding some patterns
    that will be later submitted to a custom acceptance criterion.

    It can also be used to simply discover the set of **closed itemsets** from
    a transactional dataset.

    Parameters
    ----------
    min_supp: int or float, default=0.2
        The minimum support for itemsets to be rendered in the output
        Either an int representing the absolute support, or a float for relative support
        Default to 0.2 (20%)

    n_jobs : int, default=1 The number of jobs to use for the computation. Each single item is attributed a job to
        discover potential itemsets, considering this item as a root in the search space. **Processes are preferred**
        over threads. **Carefully adjust the number of jobs** otherwise the results may be corrupted especially if you
        have the following warning: UserWarning: A worker stopped while some jobs were given to the executor.

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
    >>> patterns = lcm.fit_discover(chess)
    >>> patterns.head()
        itemset  support
    0      [58]     3195
    1      [52]     3185
    2  [52, 58]     3184
    3      [29]     3181
    4  [29, 58]     3180
    >>> patterns[patterns.itemset.map(len) > 3]  # doctest: +SKIP
    """

    def __init__(self, *, min_supp=0.2, n_jobs=1, verbose=False):
        _check_min_supp(min_supp)
        self.min_supp = min_supp  # cf docstring: minimum support provided by user
        self.max_length = -1  # maximum length of an itemset
        self.verbose = verbose
        self._min_supp = _check_min_supp(self.min_supp)
        self.item_to_tids = SortedDict()  # Dict : key item ordered by decreasing frequency , value : tids for this
        # item
        self.ord_item_freq = []  # list of ordered item by decreasing frequency
        self.n_jobs = n_jobs  # number of jobs launched by joblib
        self.lexicographic_order = False  # if true, the items of each itemset are returned in lexicographical order
        self.return_tids = False

    def fit(self, D, y=None):
        """
        fit LCM on the transactional database, by keeping records of singular items
        and their transaction ids.

        Parameters
        ----------
        D: pd.Series or iterable
            a transactional database. All entries in this D should be lists.
            If D is a pandas.Series, then `(D.map(type) == list).all()` should return `True`

        y: Ignored
            Not used, present here for API consistency by convention.

        Raises
        ------
        TypeError
            if any entry in D is not iterable itself OR if any item is not **hashable**
            OR if all items are not **comparable** with each other.
        """
        n_transactions_ = 0
        item_to_tids_ = defaultdict(Bitmap)
        for transaction in D:
            for item in transaction:
                item_to_tids_[item].add(n_transactions_)
            n_transactions_ += 1

        if isinstance(self.min_supp, float):  # make support absolute if needed
            self._min_supp = self.min_supp * n_transactions_

        low_supp_items = [k for k, v in item_to_tids_.items() if len(v) < self._min_supp]
        for item in low_supp_items:  # drop low freq items
            del item_to_tids_[item]

        ord_freq_list = sorted(item_to_tids_.items(), key=lambda item: len(item[1]), reverse=True)
        ord_freq_dic = defaultdict(Bitmap)
        ord_item_freq = []
        for idx, element in enumerate(ord_freq_list):
            item, tid = element
            ord_item_freq.append(item)
            ord_freq_dic[idx] = tid  # rename most frequent item like cat by 0, second  dog by 1

        self.item_to_tids = SortedDict(ord_freq_dic)  # {0:tids0, 1:tids1 ....}
        self.ord_item_freq = ord_item_freq  # [cat, dog, '0', ...]

        return self  # for call .fit().discover() in DiscovererMixin()

    def discover(self, *, return_tids=False, lexicographic_order=True, max_length=-1, out=None):
        """Return the set of closed itemsets, with respect to the minimum support

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        return_tids: bool, default=False
            Either to return transaction ids along with itemset.
            Default to False, will return supports instead

        lexicographic_order: bool, default=True
            Either the order of the items in each itemset is not ordered or the items are ordered lexicographically

        max_length: int, default=-1 Maximum length of an itemset. By default, -1 means that LCM returns itemsets of
        all lengths.

        out : str, default=None
            File where results are written. Discover return None. The 'out' option is usefull 
            to save memory : Instead of store all branch of lcm-tree in memory , each root 
            branch of lcm is written in a separated file in dir (TEMP_dir), and all files are
            concatenated in the final 'out' file.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  =================================
                itemset     a `list` of co-occured items
                support     frequence for this itemset
                ==========  =================================

            if `return_tids=True` then
                ==========  =================================
                itemset     a `tuple` of co-occured items
                support     frequence for this itemset
                tids        a bitmap tracking positions
                ==========  =================================

        Example
        -------
        >>> from skmine.itemsets import LCM
        >>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        >>> LCM(min_supp=2).fit_discover(D, lexicographic_order=True)
             itemset  support
        0     [2, 5]        3
        1  [2, 3, 5]        2
        >>> LCM(min_supp=2).fit_discover(D, return_tids=True)
             itemset  support       tids
        0     [2, 5]        3  (0, 1, 2)
        1  [2, 3, 5]        2     (0, 1)
        """
        self.lexicographic_order = lexicographic_order
        self.return_tids = return_tids
        self.max_length = max_length
        if out is None:  # store results in memory
            dfs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self._explore_root)(item, tids, root_file=None) for item, tids in
                list(self.item_to_tids.items())
            )  # dfs is a list of dataframe
            # make sure we have something to concat
            columns = ["itemset", "support"] if not self.return_tids else ["itemset", "support", "tids"]
            df = pd.concat([pd.DataFrame(columns=columns)] + dfs, axis=0, ignore_index=True)
            df["support"] = pd.to_numeric(df["support"])

            return df

        else:  # store results in files
            temp_dir = 'TEMP_dir'  # temporary dir where root items branch files are written
            if os.path.exists(temp_dir):  # remove dir TEMP_dir if it exists
                shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)  # create dir TEMP_dir

            Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self._explore_root)(
                    item, tids, root_file=f"{temp_dir}/root{k}.dat") for k, (
                    item, tids) in enumerate(list(self.item_to_tids.items()))
            )

            with open(out, 'w') as outfile:  # concatenate all itemsroot files located in temp_dir in a single file
                for fname in [f"{temp_dir}/root{k}.dat" for k in
                              range(len(list(self.item_to_tids.items())))]:  # all items root files
                    with open(fname) as infile:
                        for line in infile:
                            if line.strip():  # to skip empty lines
                                outfile.write(line)
            shutil.rmtree(temp_dir)  # remove the temporary dir where root files are written

            return None

    def _explore_root(self, item, tids, root_file=None):
        it = self._inner((frozenset(), tids), item)
        columns = ["itemset", "support"] if not self.return_tids else ["itemset", "support", "tids"]
        df = pd.DataFrame(data=it, columns=columns)

        if self.verbose and not df.empty:
            print("LCM found {} new itemsets from root item : {}".format(len(df), item))

        if root_file is not None:  # for writing the items root files in dir self.temp_dir

            if os.path.exists(root_file):  # delete the root file if it already exists
                os.remove(root_file)

            self.write_df_tofile(root_file, df)
            return None
        else:
            return df

    def _inner(self, p_tids, limit):

        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.item_to_tids.items())
            if tids.issubset(ids)
            if item not in p
        )

        # items are in reverse order, so the first consumed is the max
        max_k = next(takewhile(lambda e: e >= limit, cp), None)

        if max_k is not None and max_k == limit:
            p_prime = (p | set(cp) | {max_k})  # max_k has been consumed when calling next()
            # sorted items in ouput for better reproducibility
            itemset = [self.ord_item_freq[ind] for ind in list(p_prime)]
            itemset = sorted(itemset) if self.lexicographic_order else itemset

            if len(itemset) <= self.max_length or self.max_length == -1:
                if not self.return_tids:
                    yield itemset, len(tids)
                else:
                    yield itemset, len(tids), tids

            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                if tids.intersection_cardinality(ids) >= self._min_supp:
                    # new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit)

    def write_df_tofile(self, filename, df):
        with open(filename, 'w') as fw:  # write the items root files
            for index, row in df.iterrows():
                fw.write(f"({row['support']}) {' '.join(map(str, row['itemset']))}\n")
                if self.return_tids:
                    fw.write(f"{' '.join(map(str, row['tids']))}\n")


class LCMMax(LCM):
    """
    Linear time Closed item set Miner adapted to Maximal itemsets (or borders).

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
        **Processes are preferred** over threads.

    See Also
    --------
    LCM
    """

    def _inner(self, p_tids, limit):
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.item_to_tids.items())
            if tids.issubset(ids)
            if item not in p
        )
        max_k = next(cp, None)  # items are in reverse order, so the first consumed is the max

        if max_k is not None and max_k == limit:
            p_prime = (p | set(cp) | {max_k})  # max_k has been consumed when calling next()
            candidates = self.item_to_tids.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            no_cand = True

            for new_limit in candidates:
                ids = self.item_to_tids[new_limit]
                if tids.intersection_cardinality(ids) >= self._min_supp:
                    no_cand = False
                    # get new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit)

            if no_cand:  # only if no child node. This is how we PRE-check for maximality
                itemset = set([self.ord_item_freq[ind] for ind in p_prime])

                if not self.return_tids:
                    yield itemset, len(tids)
                else:
                    yield itemset, len(tids), tids

    def discover(self, *args, **kwargs):
        outfile = kwargs.get('out')
        kwargs['out'] = None
        patterns = super().discover(**kwargs)
        # keep only maximal itemsets
        maximals = filter_maximal(patterns["itemset"])
        patterns = patterns[patterns.itemset.isin(maximals)].copy()

        # keeps only itemsets smaller than max_length
        if self.max_length != -1:
            length_mask = patterns.itemset.apply(lambda x: len(x) <= self.max_length)
            patterns = patterns[length_mask]
            patterns.reset_index(drop=True, inplace=True)

        # return a list of itemset sorted or not
        patterns.loc[:, "itemset"] = patterns["itemset"].map(
            lambda i: sorted(list(i)) if self.lexicographic_order else list(i))

        if outfile:
            self.write_df_tofile(outfile, patterns)
            return None
        else:
            return patterns

    setattr(discover, "__doc__", LCM.discover.__doc__.replace("closed", "maximal"))
    setattr(discover, "__doc__", LCM.discover.__doc__.split("Example")[0])
