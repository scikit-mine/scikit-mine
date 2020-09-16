"""
Generate samples of synthetic datasets.
Mainly for benchmarks and experiments
"""

import numpy as np
import pandas as pd


def make_transactions(
    n_transactions=1000, n_items=100, density=0.5, random_state=None, item_start=0
):
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
    >>> make_transactions(n_transactions=5, n_items=20, density=.25)  # doctest: +SKIP
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
    if not 0.0 < density < 1.0:
        raise ValueError("density should be a float value between 0 and 1")

    avg_transaction_size = density * n_items

    generator = np.random.RandomState(random_state)  # pylint: disable= no-member

    item_stop = item_start + n_items
    choices = np.arange(start=item_start, stop=item_stop)
    t_sizes = generator.binomial(
        n=avg_transaction_size * 2,
        p=0.5,  # centered around avg_transaction_size
        size=n_transactions,
    )
    max_size = t_sizes.max()
    if max_size > n_items:
        delta = max_size - n_items
        t_sizes = np.clip(t_sizes, a_min=t_sizes.min() + delta, a_max=n_items)

    D = [generator.choice(choices, size, replace=False) for size in t_sizes]
    return pd.Series(D)


def make_classification(
    n_samples=100,
    n_items_per_class=100,
    *,  # pylint: disable= too-many-locals
    n_classes=2,
    weights=None,
    class_sep=0.2,
    shuffle=True,
    random_state=None,
    densities=None
):
    """
    Generate a random n-class classification problem

    Acts like sklearn version of make_classification, but produces
    transactional data instead. Transactions are drawn from a ``n_items_per_class``
    number of items, respecting the ``class_sep`` parameter to ensure transactions
    are drawn from different alphabets for different classes.

    A ``class_sep`` value of 0.0 will result in transactions being drawn from the
    same set of symbols.

    Densities can be defined for each class given the ``densities``
    parameter.

    Parameters
    ----------
    n_samples: int, default=100
        The number of samples
    n_items_per_class: int, default=100
        The number of items per class. This is similar to the ``n_features``
        parameters in scikit-learn, but operates at a class level.
    n_classes: int, default=2
        The number of classes (or labels) of the classification problem
    weigths, array-like of shape (n_classes,) default=None
        The proportions of samples assigned to each class. If None, then classes are balanced
    class_sep: float, default=0.2
        The factor of different items in different between classes.
        Setting this to 1.0 will make classification dummy.
    shuffle: boolean, default=True
        Shuffle the samples and the labels
    random_state: int RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    D: pd.Series of shape [n_samples, ]
        The generated samples
    y: pd.Series of shape [n_samples]
        Labels associated to D

    See also
    --------
    make_transactions : which is used internally to generate samples
    """
    assert n_classes > 0
    if densities is None:
        densities = [0.5] * n_classes

    if weights is None:
        weights = [1 / n_classes] * n_classes  # balanced by default

    assert len(weights) == len(densities) == n_classes
    assert 0 <= class_sep <= 1.0
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=2)

    res = dict()

    padding = 0
    for _class in range(n_classes):
        _n_samples = int(weights[_class] * n_samples)
        density = densities[_class]
        transactions = make_transactions(
            n_transactions=_n_samples,
            n_items=n_items_per_class,
            random_state=random_state,
            item_start=padding,
            density=density,
        )
        res[_class] = transactions

        # if class_sep == 1.0, then separation is strict
        padding += int(n_items_per_class - (n_items_per_class * (1 - class_sep)))

    dfs = list()
    for _class, transactions in res.items():
        df = transactions.to_frame(name="transaction")
        df.loc[:, "class"] = _class
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    if shuffle:
        df = df.sample(frac=1)

    return df["transaction"], df["class"]
