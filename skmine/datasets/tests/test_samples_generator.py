import pytest

from .._samples_generator import make_transactions
from .._samples_generator import make_classification
from ..utils import describe
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


def test_make_transactions_high_density():
    D = make_transactions(100, 10, .99)
    assert D.map(len).mean() <= 10


def test_make_transactions_wrong_value():
    with pytest.raises(ValueError):
        make_transactions(100, 10, 42)

def test_make_transactions_bilateral():
    """
    check consistency by generating a transactional dataset with given properties
    and retrieving these properties in the post-hoc description of this dataset
    """
    properties = dict(n_transactions=150, n_items=30, density=.5)
    D = make_transactions(**properties)
    desc = describe(D)
    desc.pop('avg_transaction_size')
    assert desc == pytest.approx(properties, abs=1.0)


def test_make_transactions_random_state():
    """hard setting of ``random_state`` should make two generated datasets really close"""
    properties = dict(n_transactions=150, n_items=30, density=.2, random_state=2)
    D1 = make_transactions(**properties)
    D2 = make_transactions(**properties)
    desc1 = describe(D1)
    desc2 = describe(D2)
    assert desc1 == pytest.approx(desc2, abs=.1)

def test_make_classification_wrong_value():
    with pytest.raises(ValueError):
        densities = [1.3, .2]
        make_classification(densities=densities)


def test_make_classification():
    D, y = make_classification(
        n_classes=2,
        n_items_per_class=100,
        n_samples=100,
        weights=[.3, .7],
        class_sep=.2,
    )

    desc = describe(D)
    full_desc = dict(
        n_items=120,  # = n_items_perclass * (1 + class_sep)
        avg_transaction_size=50,
        n_transactions=100,
        density=.40,  # both have .5 densities, but only 20% of it is intersecting, see class_sep
    )
    assert describe(D) == pytest.approx(full_desc, abs=1)
