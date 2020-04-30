import pytest

from .._samples_generator import make_transactions
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
