"""
utils methods for skmine.datasets
"""

from itertools import chain


def describe_itemsets(D):
    """
    Parameters
    ----------
    D: pd.Series
        A transactional dataset

    Example
    -------
    >>> from skmine.datasets.fimi import fetch_connect
    >>> from skmine.datasets.utils import describe_itemsets
    >>> describe_itemsets(fetch_connect())
    {'n_items': 75, 'avg_transaction_size': 37.0, 'n_transactions': 3196}
    """
    return dict(
        n_items=len(set(chain(*D))),
        avg_transaction_size=D.map(len).mean(),
        n_transactions=D.shape[0]
    )
