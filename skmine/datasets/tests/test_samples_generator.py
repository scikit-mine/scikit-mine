import pytest

from .._samples_generator import make_transactions
from ..utils import describe_itemsets
from itertools import chain
import numpy as np

def test_make_transactions():
    n_transactions = 100
    n_items = 20
    avg_transaction_size = 10
    D = make_transactions(n_transactions, n_items, avg_transaction_size)
    assert len(D) == n_transactions
    assert set(chain(*D)) == set(range(n_items))

    lens = D.map(len)
    assert lens.max() < n_items
    np.testing.assert_almost_equal(lens.mean(), avg_transaction_size, 0)

def test_bilateral():
    """
    check consistency by generating a transactional dataset with given properties
    and retrieving these properties in the post-hoc description of this dataset
    """
    properties = dict(
        n_transactions=150,
        n_items=30,
        avg_transaction_size=20,
    )
    D = make_transactions(**properties)
    desc = describe_itemsets(D)
    assert desc['n_transactions'] == properties['n_transactions']
    assert desc['n_items'] == properties['n_items']
    assert round(desc['avg_transaction_size']) == properties['avg_transaction_size']
