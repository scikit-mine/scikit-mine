import pandas as pd

from ..utils import describe


def test_describe():
    D = pd.Series([[2, 3, 4], [10, 3]])
    desc = describe(D)
    assert isinstance(desc, dict)
    assert desc == {
        "n_items": 4,
        "avg_transaction_size": 2.5,
        "n_transactions": 2,
        "density": 0.625,
    }
