import pytest

from ..utils import describe_itemsets
import pandas as pd

def test_describe():
    D = pd.Series([[2, 3, 4], [10, 3]])
    desc = describe_itemsets(D)
    assert isinstance(desc, dict)
    assert desc == {
        'nb_items': 4,
        'avg_transaction_size': 2.5,
        'nb_transactions': 2,
    }
