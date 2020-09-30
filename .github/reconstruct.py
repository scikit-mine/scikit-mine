"""
Python script to ensure full reconstruction of datasets using MDL miners

As MDL is a lossless compression framework, the entire original data should be reconstructed from
the concise representation that MDL provides
"""

from skmine.itemsets import SLIM
from skmine.datasets.fimi import fetch_mushroom
from skmine.datasets.fimi import fetch_chess
from skmine.bitmaps import Bitmap
from skmine.preprocessing import TransactionEncoder
from skmine.itemsets.slim import update_usages

import pandas as pd

import inspect


def reconstruct(codetable):
    """
    Parameters
    ----------
    codetable: pd.Series
        a codetable, mapping itemsets to their usage tids
    """
    n_transactions = codetable.map(Bitmap.max).max() + 1

    D = pd.Series([set()] * n_transactions)

    for itemset, tids in codetable.iteritems():
        D.iloc[list(tids)] = D.iloc[list(tids)].map(itemset.union)
    return D.map(sorted)


if __name__ == "__main__":
    print(inspect.getsource(update_usages))
    Ds = [fetch_mushroom(), fetch_chess()]
    for D in Ds:
        slim = SLIM(pruning=False, n_iter_no_change=1000)
        _D = TransactionEncoder().fit_transform(D)
        slim.fit(_D)
        r_D = reconstruct(slim.codetable)
        pd.testing.assert_series_equal(D, r_D, check_names=False)
