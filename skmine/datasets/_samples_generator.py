"""
Generate samples of synthetic datasets.
Mainly for benchmarks and experiments
"""

import random

import numpy as np
import pandas as pd


def make_transactions(n_transactions=1000,
                      n_items=100,
                      avg_transaction_size=10):
    """
    Generate a transactional dataset with predefined properties

    see: https://liris.cnrs.fr/Documents/Liris-3716.pdf

    Transaction sizes follow a normal distribution, centered around
    ``avg_transaction_size``.
    Individual items are integer values between 0 and ``n_items``.

    Parameters
    ---------
    n_transactions: int, default=1000
        The number of transactions to generate
    n_items: int, default=100
        The number of indidual items, i.e the size of the set of symbols
    avg_transaction_size: int, default=10
        The average size for a transactions

    References
    ----------
    .. [1] F. Flouvat, F. De Marchi, JM. Petit
           "A new classification of datasets for frequent itemsets", 2009

    Returns
    -------
    pd.Series: a Series of shape (``n_transactions``,)
        Earch entry is a list of integer values
    """
    choices = list(range(n_items))
    t_sizes = np.random.binomial(
        n=int(avg_transaction_size * 2),
        p=.5,  # centered around avg_transaction_size
        size=n_transactions
    )
    transactions = [random.choices(choices, k=size) for size in t_sizes]
    return pd.Series(transactions)
