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


def describe_logs(D):
    """Give some high level properties on logs

    ==============================      =====
    Number of events                    int
    Average delta per event             float
    Average nb of points per event      float
    ==============================      =====

    Parameters
    ----------
    D: pd.Series
        A dataset containing logs

    Example
    -------
    >>> from skmine.datasets.periodic import fetch_health_app
    >>> from skmine.datasets.utils import describe_logs
    >>> describe(fetch_health_app()) # doctest: +SKIP
    {'n_events': 20,
    'avg_delta_per_event': Timedelta('0 days 00:53:24.984000'),
    'avg_nb_points_per_event': 100.0}
    """
    gb = D.groupby(D.values)
    a = gb.apply(lambda df: (df.index.max() - df.index.min(), len(df)))
    avg_nb_points = a.str[1].mean()

    return dict(
        n_events=len(gb),
        avg_delta_per_event=a.str[0].mean(),
        avg_nb_points_per_event=avg_nb_points,
    )
