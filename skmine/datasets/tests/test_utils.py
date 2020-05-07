import pytest

from ..utils import describe_transactions
import pandas as pd

def test_describe():
    D = pd.Series([[2, 3, 4], [10, 3]])
    desc = describe_transactions(D)
    assert isinstance(desc, dict)
    assert desc == {
        'n_items': 4,
        'avg_transaction_size': 2.5,
        'n_transactions': 2,
    }
