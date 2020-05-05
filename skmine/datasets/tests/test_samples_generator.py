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
    properties = dict(n_transactions=150, n_items=30, avg_transaction_size=15)
    D = make_transactions(**properties)
    desc = describe_itemsets(D)
    assert desc == pytest.approx(properties, abs=1.0)


def test_bilateral_n_items_too_low():
    """
    If n_items if too low to fill transactions w.r.t avg_transaction_size,
    we automatically adapt
    """
    properties = dict(n_transactions=150, n_items=30, avg_transaction_size=20)
    with pytest.raises(ValueError):
        D = make_transactions(**properties)



def test_random_state():
    """hard setting of ``random_state`` should make two generated datasets really close"""
    properties = dict(n_transactions=150, n_items=30, avg_transaction_size=15, random_state=2)
    D1 = make_transactions(**properties)
    D2 = make_transactions(**properties)
    desc1 = describe_itemsets(D1)
    desc2 = describe_itemsets(D2)
    assert desc1 == pytest.approx(desc2, abs=.1)