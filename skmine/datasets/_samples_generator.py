"""
Generate samples of synthetic datasets.
Mainly for benchmarks and experiments
"""

import numpy as np
import pandas as pd


def make_transactions(n_transactions=1000,
                      n_items=100,
                      density=.5,
                      random_state=None):
    """
    Generate a transactional dataset with predefined properties

    see: https://liris.cnrs.fr/Documents/Liris-3716.pdf

    Transaction sizes follow a normal distribution, centered around ``density * n_items``.
    Individual items are integer values between 0 and ``n_items``.

    Parameters
    ---------
    n_transactions: int, default=1000
        The number of transactions to generate
    n_items: int, default=100
        The number of indidual items, i.e the size of the set of symbols
    density: float, default=0.5
        Density of the resulting dataset
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    References
    ----------
    .. [1] F. Flouvat, F. De Marchi, JM. Petit
           "A new classification of datasets for frequent itemsets", 2009

    Example
    -------
    >>> from skmine.datasets import make_transactions
    >>> make_transactions(n_transactions=5, n_items=20, density=.25)
    0    [0, 6, 18, 10, 1, 12]
    1          [2, 18, 10, 14]
    2                [4, 5, 1]
    3         [10, 11, 16, 19]
    4     [9, 4, 19, 8, 12, 5]
    dtype: object

    Notes
    -----
    With a binary matrix representation of the resulting dataset, we have the following equality
        .. math:: density = { Number\ of\ ones \over Number\ of\ cells }
    This is equivalent to
        .. math:: density = { Average\ transaction\ size \over number\ of\ items }

    Returns
    -------
    pd.Series: a Series of shape (``n_transactions``,)
        Earch entry is a list of integer values
    """
    if not .0 < density < 1.0:
        raise ValueError('density should be a float value between 0 and 1')
    avg_transaction_size = density * n_items

    generator = np.random.RandomState(random_state)  # pylint: disable= no-member

    choices = np.arange(n_items)
    t_sizes = generator.binomial(
        n=avg_transaction_size * 2,
        p=.5,  # centered around avg_transaction_size
        size=n_transactions
    )
    max_size = t_sizes.max()
    if max_size > n_items:
        delta = max_size - n_items
        t_sizes = np.clip(t_sizes, a_min=t_sizes.min() + delta, a_max=n_items)

    D = [generator.choice(choices, size, replace=False) for size in t_sizes]
    return pd.Series(D)
