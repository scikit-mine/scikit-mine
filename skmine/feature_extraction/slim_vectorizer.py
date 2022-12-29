"""
SLIM vectorizer, using SLIM as a feature extraction scheme
"""

import numpy as np
import pandas as pd

from ..base import TransformerMixin
from ..itemsets import SLIM
from ..itemsets.slim import _to_vertical, cover

STRATEGIES = ("codes", "one-hot")


def _filter_stop_items(D, stop_items):
    for t in D:
        yield set(t).difference(stop_items)


class SLIMVectorizer(SLIM, TransformerMixin):
    """SLIM mining, turned into a feature extraction step for sklearn

    `k` new itemsets (associations of one or more items) are learned at training time

    The model (pattern set) is then used to cover new data, in order of usage.
    This is similar to one-hot-encoding, except the dimension will be much more concise,
    because the columns will be patterns learned via an MDL criterion.

    Parameters
    ----------
    strategy: str, default="codes"
        If the chosen strategy is set to `one-hot`, non-zero cells are filled with ones.

        If the chosen `strategy` is left to `codes`, non-zero cells are filled with code lengths,
        i.e the probabity of the pattern in the training data.

    k: int, default=5
        Number of non-singleton itemsets to mine.
        A singleton is an itemset containing a single item.

        Calls to `.transform` will output pandas.DataFrame with `k` columns

    pruning: bool, default=False
        Either to activate pruning or not.

    stop_items: iterable, default=None
        Set of items to filter out while ingesting the input data.


    Examples
    --------
    >>> from skmine.feature_extraction import SLIMVectorizer
    >>> D = [['bananas', 'milk'], ['milk', 'bananas', 'cookies'], ['cookies', 'butter', 'tea']]
    >>> SLIMVectorizer(k=2).fit_transform(D)
       (bananas, milk)  (cookies,)
    0              0.4         0.0
    1              0.4         0.4
    2              0.0         0.4


    See Also
    --------
    skmine.itemsets.SLIM

    Notes
    -----
    This transformer does not output scipy.sparse matrices,
    as SLIM should learn a concise description of the data,
    and covering new data with this small set of high usage
    patterns should output matrices with very few zeros.
    """

    def __init__(self, strategy="codes", *, k=5, pruning=False, stop_items=None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.pruning = pruning
        self.stop_items = stop_items
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.strategy = strategy

    def fit(self, D, y=None):
        if self.stop_items is not None:
            D = _filter_stop_items(D, stop_items=self.stop_items)
        return super().fit(D)

    def transform(self, D, y=None):
        """Transform new data

        Parameters
        ----------
        D: iterable
            transactional data

        Returns
        -------
        pd.DataFrame
            a dataframe of `len(D)` rows and `self.k` columns

        See Also
        --------
        skmine.itemsets.SLIM.cover
        """
        stop_items = self.stop_items or set()
        D_sct, _len = _to_vertical(D, stop_items=stop_items, return_len=True)

        code_lengths = self.discover(return_tids=False, singletons=True, drop_null_usage=False)
        code_lengths = pd.Series(code_lengths["usage"].values, index=code_lengths["itemset"].apply(tuple))
        code_lengths = code_lengths[code_lengths.index.map(set(D_sct).issuperset)]

        isets = code_lengths.nlargest(self.k)  # real usages sorted in decreasing order
        covers = cover(D_sct, isets.index)

        mat = np.zeros(shape=(_len, len(covers)))
        for idx, tids in enumerate(covers.values()):
            mat[tids, idx] = 1
        mat = pd.DataFrame(mat, columns=list(covers.keys()))

        if self.strategy == "codes":
            ct_codes = code_lengths[isets.index] / code_lengths.sum()
            mat = (mat * ct_codes).astype(np.float32)

        return mat
