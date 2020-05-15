"""
utils methods for skmine.datasets
"""

from itertools import chain


def describe(D):
    """Give some high level properties on transactions

    ========================    ===============
    Number of items             int
    Number of transactions      int
    Average transaction size    float
    Density                     float in [0, 1]
    ========================    ===============

    Parameters
    ----------
    D: pd.Series
        A transactional dataset

    Notes
    -----
        .. math:: density = { avg\_transaction\_size \over n\_items }

    Example
    -------
    >>> from skmine.datasets.fimi import fetch_connect
    >>> from skmine.datasets.utils import describe
    >>> describe(fetch_connect())  # doctest: +SKIP
    {'n_items': 75, 'avg_transaction_size': 37.0, 'n_transactions': 3196, 'density': 0.4933}
    """
    avg_transaction_size = D.map(len).mean()
    n_transactions = D.shape[0]
    n_items = len(set(chain(*D)))
    return dict(
        n_items=n_items,
        avg_transaction_size=avg_transaction_size,
        n_transactions=n_transactions,
        density=avg_transaction_size / n_items,
    )
