import pytest

from .._samples_generator import make_transactions
from ..utils import describe_transactions
from itertools import chain
import numpy as np

def test_make_transactions():
    n_transactions = 100
    n_items = 20
    avg_transaction_size = 10
    D = make_transactions(n_transactions, n_items, .5)
    assert len(D) == n_transactions
    assert set(chain(*D)) == set(range(n_items))

    lens = D.map(len)
    assert lens.max() < n_items
    np.testing.assert_almost_equal(lens.mean(), avg_transaction_size, 0)


def test_high_density():
    D = make_transactions(100, 10, .99)
    assert D.map(len).mean() <= 10


def test_wrong_value():
    with pytest.raises(ValueError):
        make_transactions(100, 10, 42)

def test_bilateral():
    """
    check consistency by generating a transactional dataset with given properties
    and retrieving these properties in the post-hoc description of this dataset
    """
    properties = dict(n_transactions=150, n_items=30, density=.5)
    D = make_transactions(**properties)
    desc = describe_transactions(D)
    desc.pop('avg_transaction_size')
    assert desc == pytest.approx(properties, abs=1.0)


def test_random_state():
    """hard setting of ``random_state`` should make two generated datasets really close"""
    properties = dict(n_transactions=150, n_items=30, density=.2, random_state=2)
    D1 = make_transactions(**properties)
    D2 = make_transactions(**properties)
    desc1 = describe_transactions(D1)
    desc2 = describe_transactions(D2)
    assert desc1 == pytest.approx(desc2, abs=.1)
